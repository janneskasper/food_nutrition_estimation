import segmentation_models as sm
import keras
import numpy as np
import os
import json
from src.food_recognition_options import ModelConfig
from src.models.custom_modules import *

def getSegmentationModel(config: ModelConfig, lr=0.001):
    print("[*] Loading food segmentation model ...")

    num_classes = len(config.classes) + 1 # + 1 for adding background

    model: keras.models.Model = sm.Unet(config.model_backbone, 
                                        encoder_weights="imagenet", 
                                        # input_shape=options.input_size, 
                                        classes=num_classes)
    
    dice_loss = sm.losses.DiceLoss(class_indexes=np.arange(len(config.classes))) # last class is the background we want to ignore
    focal_loss = sm.losses.BinaryFocalLoss() if config.classes == 1 else sm.losses.CategoricalFocalLoss(class_indexes=np.arange(len(config.classes)))
    total_loss = dice_loss + focal_loss

    metrics = [sm.metrics.IOUScore(threshold=0.5, class_indexes=np.arange(len(config.classes))), 
               sm.metrics.FScore(threshold=0.5, class_indexes=np.arange(len(config.classes)))]
    
    optim = keras.optimizers.Adam(lr=lr)

    if config.model_weights_path is not None:
        weights_path = os.path.join(os.getcwd(), config.model_weights_path)
        assert os.path.isfile(weights_path), f"Loading segmentation model weights: file not found!"
        model.load_weights(weights_path)

    model.compile(optimizer=optim, loss=total_loss, metrics=metrics)
    return model

def getDepthEstimationModel(config: ModelConfig):
    # Load depth estimation model
    print("[*] Loading depth estimation model ...")

    custom_losses = Losses()
    objs = {'ProjectionLayer': ProjectionLayer,
            'ReflectionPadding2D': ReflectionPadding2D,
            'InverseDepthNormalization': InverseDepthNormalization,
            'AugmentationLayer': AugmentationLayer,
            'compute_source_loss': custom_losses.compute_source_loss}
    
    model_path = os.path.join(os.getcwd(), config.model_path_json)
    model_weights_path = os.path.join(os.getcwd(), config.model_weights_path)

    assert os.path.isfile(model_path), f"Loading depth model: file not found!"
    assert os.path.isfile(model_weights_path), f"Loading depth model weights: file not found!"

    with open(model_path, 'r') as read_file:
        model_architecture_json = json.load(read_file)
        monovideo: keras.models.Model = keras.models.model_from_json(model_architecture_json, custom_objects=objs)
    __set_weights_trainable(monovideo, False)
    monovideo.load_weights(model_weights_path)
    depth_net = monovideo.get_layer('depth_net')
    depth_model = keras.models.Model(inputs=depth_net.inputs,
                                outputs=depth_net.outputs,
                                name='depth_model')
    return depth_model

def __set_weights_trainable(model, trainable):
    """Sets model weights to trainable/non-trainable.

    Inputs:
        model: Model to set weights.
        trainable: Trainability flag.
    """
    for layer in model.layers:
        layer.trainable = trainable
        if isinstance(layer, keras.models.Model):
            __set_weights_trainable(layer, trainable)