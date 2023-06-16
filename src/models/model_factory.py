import segmentation_models as sm
import keras
import numpy as np
import os
import json
from src.food_recognition_options import ModelConfig
from src.models.custom_modules import *

def getSegmentationModel(config: ModelConfig, lr=0.001, training=False, name_filter=None):
    print("[*] Loading food segmentation model ...")

    num_classes = len(config.classes) + 1 # + 1 for adding background

    if config.model == "unet":
        model: keras.models.Model = sm.Unet(config.model_backbone, 
                                        encoder_weights="imagenet", 
                                        input_shape=config.input_size, 
                                        classes=num_classes)
    elif config.model == "fpn":
        model: keras.models.Model = sm.FPN(config.model_backbone,
                                           encoder_weights="imagenet",
                                           input_shape=config.input_size,
                                           classes=num_classes)
    else:
        return None
    
    loss = sm.losses.CategoricalCELoss(class_indexes=np.arange(len(config.classes))) + sm.losses.DiceLoss(class_indexes=np.arange(len(config.classes)))

    metrics = [sm.metrics.IOUScore(threshold=0.5, class_indexes=np.arange(len(config.classes))), 
               sm.metrics.FScore(threshold=0.5, class_indexes=np.arange(len(config.classes)))
               ]
    
    optim = keras.optimizers.Adam(lr=lr)

    if config.model_weights_path is not None:
        weights_path = os.path.join(os.getcwd(), config.model_weights_path)
        assert os.path.isfile(weights_path), f"Loading segmentation model weights: file not found!"
        model.load_weights(weights_path)
        print("[*] Loaded weights from file ", weights_path)

    __set_weights_trainable(model, trainable=training, name_filter=name_filter)

    model.compile(optimizer=optim, loss=loss, metrics=metrics)
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

def __set_weights_trainable(model, trainable, name_filter=None):
    """Sets model weights to trainable/non-trainable.

    Inputs:
        model: Model to set weights.
        trainable: Trainability flag.
    """
    for layer in model.layers:
        if name_filter is not None:
            if  name_filter in layer.name:
                layer.trainable = trainable
            else:
                layer.trainable = not trainable
        else:
            layer.trainable = trainable
        if isinstance(layer, keras.models.Model):
            __set_weights_trainable(layer, trainable, name_filter)