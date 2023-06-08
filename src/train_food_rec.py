import time
from load_food_rec import CocoDatasetGenerator, CocoDatasetLoader, visualize
from food_recognition_options import FoodRecognitionOptions
import segmentation_models as sm
import keras
import os
import datetime
import albumentations as A
import cv2
import numpy as np
import matplotlib.pyplot as plt

CLASSES = ['water',
            'salad-leaf-salad-green',
            'bread-white',
            'tomato-raw',
            'butter', 
            'carrot-raw',
            'rice', 
            'egg', 
            'apple', 
            'jam', 
            'cucumber', 
            'banana', 
            'cheese', 

        #   'bread-wholemeal', 
        #   'coffee-with-caffeine', 
        #   'mixed-vegetables', 
        #   'wine-red', 
        #   'potatoes-steamed', 
        #   'bell-pepper-red-raw', 
        #   'hard-cheese', 
        #   'espresso-with-caffeine', 
        #   'tea', 
        #   'bread-whole-wheat', 
        #   'mixed-salad-chopped-without-sauce', 
        #   'avocado', 
        #   'white-coffee-with-caffeine', 
        #   'tomato-sauce', 
        #   'wine-white', 
        #   'broccoli', 
        #   'strawberries', 
        #   'pasta-spaghetti'
            ]

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        A.Lambda(image=preprocessing_fn),
    ]
    return A.Compose(_transform)

def train(options: FoodRecognitionOptions):
    model = getModel(options=options)
    
    preprocessor = sm.get_preprocessing(options.backbone)
    train_generator = CocoDatasetLoader(CocoDatasetGenerator(batch_size=options.batch_size,
                                           preprocessor=get_preprocessing(preprocessor), 
                                           annotations_path=options.train_ann_path, 
                                           img_dir=options.train_img_path, 
                                           data_size=options.input_size,
                                           filter_categories=CLASSES
                                           ))
    val_generator = CocoDatasetLoader(CocoDatasetGenerator(batch_size=1,
                                         preprocessor=get_preprocessing(preprocessor), 
                                         annotations_path=options.val_ann_path, 
                                         img_dir=options.val_img_path, 
                                         data_size=options.input_size,
                                         filter_categories=CLASSES
                                         ))
    # class_weights = train_generator.getClassDistribution()
    # max_dist = max(class_weights.values())
    # for k,v in class_weights.items():
    #     class_weights[k] = max_dist / v
    
    t = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = os.path.join(options.log_dir, f"train_{t}")
    callbacks = [
        keras.callbacks.ModelCheckpoint(os.path.join(path, "weights_epoch_{epoch:02d}.hdf5"), verbose=1),
        keras.callbacks.TensorBoard(os.path.join(path, f"tensorboard_logs"), update_freq="batch"),
        keras.callbacks.ReduceLROnPlateau(verbose=1)
    ]

    model.fit_generator(train_generator, 
                        steps_per_epoch=options.steps_epoch, 
                        epochs=options.epochs, 
                        validation_data=val_generator, 
                        validation_steps=len(val_generator), 
                        callbacks=callbacks,
                        workers=6,
                        # use_multiprocessing=True
                        # class_weight=class_weights
                        )

    model.save(os.path.join(path, "final_model.keras"))
    keras.models.save_model(model, os.path.join(path, "final_model.hdf5"))
    model.save_weights(os.path.join(path, "final_weights_only.hdf5"))

def getModel(options: FoodRecognitionOptions):
    num_classes = len(options.seg_options.classes) + 1 # + 1 for adding background

    model: keras.models.Model = sm.Unet(options.seg_options.training_params.backbone, 
                                        encoder_weights="imagenet", 
                                        # input_shape=options.input_size, 
                                        classes=num_classes)
    
    dice_loss = sm.losses.DiceLoss(class_indexes=np.arange(len(options.seg_options.classes))) # last class is the background we want to ignore
    focal_loss = sm.losses.BinaryFocalLoss() if num_classes == 1 else sm.losses.CategoricalFocalLoss(class_indexes=np.arange(len(options.seg_options.classes)))
    total_loss = dice_loss + focal_loss
    metrics = [sm.metrics.IOUScore(threshold=0.5, class_indexes=np.arange(len(options.seg_options.classes))), 
               sm.metrics.FScore(threshold=0.5, class_indexes=np.arange(len(options.seg_options.classes)))]
    optim = keras.optimizers.Adam(lr=options.seg_options.training_params.lr)

    if options.seg_options.training_params.model_weights_path is not None and os.path.isfile(options.seg_options.training_params.model_weights_path):
        model.load_weights(options.seg_options.training_params.model_weights_path)

    model.compile(optimizer=optim, loss=total_loss, metrics=metrics)
    return model

def testModel(model_path, options: FoodRecognitionOptions, test_images):
    assert os.path.isfile(model_path), f"Model weigths not found"
    model = getModel(options)
    model.load_weights(model_path)

    for path in test_images:
        # img = cv2.imread(os.path.join(options.val_img_path, path))
        img = cv2.imread("/home/jannes/Documents/MasterDelft/Q4/DeepLearning/datasets/filtered/apple_6.jpg")
        # img = cv2.imread(f"{os.getcwd()}/datasets/filtered/carrot-raw_17.jpg")
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, options.seg_options.training_params.input_size[:2])

        # preprocessor = get_preprocessing(sm.get_preprocessing(options.seg_options.training_params.backbone))
        # sample = preprocessor(image=img)
        # img = sample["image"]

        img = np.expand_dims(img, axis=0)
        pred = model.predict(img)[0]

        output_raw = np.argmax(pred, axis=2)
        m = np.max(output_raw)
        output = cv2.cvtColor((output_raw * (255 / m)).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        cv2.imshow("Img", img[0,...])
        cv2.imshow("Mask", output)
        cv2.imshow("Mask Raw", output_raw.astype(np.uint8))

        cv2.waitKey(0)
        # visualize(
        #     image=denormalize(img.squeeze()),
        #     pred=pred[...,:5].squeeze(),
        #     )
   
# helper function for data visualization    
def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)    
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x

def manualTrainingStep(options: FoodRecognitionOptions):
    preprocessor = sm.get_preprocessing(options.backbone)
    train_generator = CocoDatasetGenerator(batch_size=1,
                                        # preprocessor=get_preprocessing(preprocessor), 
                                        annotations_path=options.train_ann_path, 
                                        img_dir=options.train_img_path, 
                                        data_size=options.input_size,
                                        filter_categories=CLASSES
                                        )
    t = 0.0
    for i in range(100):
        t1 = time.time()
        b = next(train_generator.getGenerator())
        img = b[0][0]
        mask = b[1][0][...,0] * 255
        img = img.astype(np.uint8)
        mask = mask.astype(np.uint8)
        for i in range(1,30):
            print(np.any(b[1][0][...,i]))
        cv2.imshow("Img", img)
        cv2.imshow("mask", mask)
        cv2.waitKey(0)
        t += time.time() - t1
    print(f"Avg time per next: {t/100}")
    print(f"Total time for 100 next: {t}")
    return


if __name__ == "__main__":
    # train(FoodRecognitionOptions())
    # manualTrainingStep(FoodRecognitionOptions())
    testModel("/home/jannes/Documents/MasterDelft/Q4/DeepLearning/food_volume_estimation/nutrition_estimation/models/files/seg_model_e18.hdf5",
              FoodRecognitionOptions(),
              ["111631.jpg"])


