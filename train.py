from src.load_food_rec import CocoDatasetGenerator, CocoDatasetLoader
from src.food_recognition_options import TrainingConfig
from src.models.model_factory import getSegmentationModel
import segmentation_models as sm
import keras
import keras.backend as K
import tensorflow as tf
import os
import datetime
import albumentations as A
import numpy as np
import pandas as pd

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


def train(config: TrainingConfig):
    model = getSegmentationModel(config=config.model_options, lr=config.lr, training=True, name_filter="decode")
    preprocessor = sm.get_preprocessing(config.model_options.model_backbone)

    train_gen = CocoDatasetGenerator(batch_size=config.batch_size,
                                           preprocessor=get_preprocessing(preprocessor), 
                                           annotations_path=config.train_ann_path, 
                                           img_dir=config.train_img_path, 
                                           data_size=config.model_options.input_size,
                                           filter_categories=config.model_options.classes
                                           )


    val_gen = CocoDatasetGenerator(batch_size=1,
                                    preprocessor=get_preprocessing(preprocessor), 
                                    annotations_path=config.val_ann_path, 
                                    img_dir=config.val_img_path, 
                                    data_size=config.model_options.input_size,
                                    filter_categories=config.model_options.classes
                                    )
    train_loader = CocoDatasetLoader(train_gen)
    val_loader = CocoDatasetLoader(val_gen)

    config.model_options.classes = train_gen.categoryNames # the order of classes gets shuffled around when loading the coco dataset

    t = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = os.path.join(config.log_dir, f"train_{t}")
    callbacks = [
        keras.callbacks.ModelCheckpoint(os.path.join(path, "weights_epoch_{epoch:02d}.hdf5"), verbose=1, save_weights_only=True),
        keras.callbacks.TensorBoard(os.path.join(path, f"tensorboard_logs"), update_freq="batch", ),
        keras.callbacks.ReduceLROnPlateau(verbose=1),
    ]

    if config.class_weights is None:
        classes, total = train_gen.getClassDistribution(False)
        weights = [0] * len(classes.keys())
        for k,v in classes.items():
            weights[config.model_options.classes.index(k)] = total / (len(classes.keys()) * v)
        config.class_weights = weights

    os.makedirs(path)
    config.saveToJson(os.path.join(path, "config.json"))

    trainable_count = keras.utils.layer_utils.count_params(model.trainable_weights)
    non_trainable_count = keras.utils.layer_utils.count_params(model.non_trainable_weights)

    print("============================================================================")
    print(f"[*] Number of trainable params: {trainable_count}")
    print(f"[*] Number of non-trainable params: {non_trainable_count}")
    print(f"[*] Number of training steps: {len(train_loader) // config.batch_size}")
    print(f"[*] Number of validation steps: {len(val_loader)}")
    print(f"[*] Classes: {config.model_options.classes}")
    print(f"[*] Class weights: {weights}")
    print("============================================================================")

    history: keras.callbacks.History = model.fit_generator(train_loader, 
                        steps_per_epoch=len(train_loader) // config.batch_size, 
                        epochs=config.epochs, 
                        validation_data=val_loader, 
                        validation_steps=len(val_loader), 
                        callbacks=callbacks,
                        workers=config.workers,
                        use_multiprocessing=True,
                        class_weight=weights
                        )

    model.save(os.path.join(path, "final_model.keras"))
    keras.models.save_model(model, os.path.join(path, "final_model.hdf5"))
    model.save_weights(os.path.join(path, "final_weights_only.hdf5"))
    
    hist_df = pd.DataFrame(history.history) 

    with open(os.path.join(path, "history.json"), mode='w') as f:
        hist_df.to_json(f)

def test(config: TrainingConfig):
    model = getSegmentationModel(config=config.model_options, lr=config.lr)
    
    preprocessor = sm.get_preprocessing(config.model_options.model_backbone)
    train_generator = CocoDatasetLoader(CocoDatasetGenerator(batch_size=1,
                                           preprocessor=get_preprocessing(preprocessor), 
                                           annotations_path=config.train_ann_path, 
                                           img_dir=config.train_img_path, 
                                           data_size=config.model_options.input_size,
                                           filter_categories=config.model_options.classes
                                           ))
    img, gt = train_generator[0]

    r = model.predict(img)
    
    t = tf.convert_to_tensor(gt, dtype=tf.float32)
    r = tf.convert_to_tensor(r, dtype=tf.float32)
    loss = sm.losses.DiceLoss()
    loss = loss + sm.losses.CategoricalFocalLoss()
    res = loss(t, r)

    print(res)


if __name__ == "__main__":
    train(TrainingConfig.createTrainConfig())


