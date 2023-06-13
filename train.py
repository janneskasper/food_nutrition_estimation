from src.load_food_rec import CocoDatasetGenerator, CocoDatasetLoader
from src.food_recognition_options import TrainingConfig
from src.models.model_factory import getSegmentationModel
import segmentation_models as sm
import keras
import os
import datetime
import albumentations as A


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


def train(options: TrainingConfig):
    model = getSegmentationModel(config=options.model_options, lr=options.lr)
    
    preprocessor = sm.get_preprocessing(options.model_options.model_backbone)
    train_generator = CocoDatasetLoader(CocoDatasetGenerator(batch_size=options.batch_size,
                                           preprocessor=get_preprocessing(preprocessor), 
                                           annotations_path=options.train_ann_path, 
                                           img_dir=options.train_img_path, 
                                           data_size=options.model_options.input_size,
                                           filter_categories=options.model_options.classes
                                           ))
    val_generator = CocoDatasetLoader(CocoDatasetGenerator(batch_size=1,
                                         preprocessor=get_preprocessing(preprocessor), 
                                         annotations_path=options.val_ann_path, 
                                         img_dir=options.val_img_path, 
                                         data_size=options.model_options.input_size,
                                         filter_categories=options.model_options.classes
                                         ))
    
    t = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = os.path.join(options.log_dir, f"train_{t}")
    callbacks = [
        keras.callbacks.ModelCheckpoint(os.path.join(path, "weights_epoch_{epoch:02d}.hdf5"), verbose=1, save_weights_only=True),
        keras.callbacks.TensorBoard(os.path.join(path, f"tensorboard_logs"), update_freq="batch"),
        keras.callbacks.ReduceLROnPlateau(verbose=1)
    ]

    model.fit_generator(train_generator, 
                        steps_per_epoch=options.steps_epoch, 
                        epochs=options.epochs, 
                        validation_data=val_generator, 
                        validation_steps=len(val_generator), 
                        callbacks=callbacks,
                        workers=options.workers,
                        )

    model.save(os.path.join(path, "final_model.keras"))
    keras.models.save_model(model, os.path.join(path, "final_model.hdf5"))
    model.save_weights(os.path.join(path, "final_weights_only.hdf5"))


if __name__ == "__main__":
    train(TrainingConfig.createTrainConfig())


