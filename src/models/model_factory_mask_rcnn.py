import segmentation_models as sm
import keras
import numpy as np
import os
import json
from src.food_recognition_options import FoodRecognitionOptions
from src.models.custom_modules import *

from mrcnn.config import Config
from mrcnn import (
    model as modellib, 
    utils)

DEFAULT_LOGS_DIR = 'logs/'

class FoodConfig(Config):
    """Configuration for training on the UNIMIB2016 food dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = 'Food_segmentation'

    # Adjust to appropriate GPU specifications
    GPU_COUNT = 1

    # # Try one 20230610T2213
    IMAGES_PER_GPU = 1

    
    # # Try two 20230610T2251
    IMAGES_PER_GPU = 3
    BACKBONE = 'resnet18'


    # # Number of classes (including background)
    # NUM_CLASSES = 1 + len(clusters)

    # Otherwise batch_gt_class_ids[b, :gt_class_ids.shape[0]] = gt_class_ids will crash
    MAX_GT_INSTANCES = 200

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 3000 // IMAGES_PER_GPU

    # Validation steps per epoch
    VALIDATION_STEPS = 300 // IMAGES_PER_GPU

    # Less ROIs to keep 1:3 ratio
    TRAIN_ROIS_PER_IMAGE = 128

    def __init__(self, num_classes):
        self.NUM_CLASSES = 1 + num_classes
        super().__init__()

def getSegmentationModel(options: FoodRecognitionOptions):
    print("Loading food segmentation model ...")

    num_classes = len(options.seg_options.classes) + 1 # + 1 for adding background

    # model: keras.models.Model = sm.Unet(options.seg_options.training_params.backbone, 
    #                                     encoder_weights="imagenet", 
    #                                     # input_shape=options.input_size, 
    #                                     classes=num_classes)
    model: keras.models.Model = modellib.MaskRCNN(mode='training', config=FoodConfig(len(options.seg_options.classes)),
                                  model_dir=DEFAULT_LOGS_DIR)
    
    dice_loss = sm.losses.DiceLoss(class_indexes=np.arange(len(options.seg_options.classes))) # last class is the background we want to ignore
    focal_loss = sm.losses.BinaryFocalLoss() if options.seg_options.classes == 1 else sm.losses.CategoricalFocalLoss(class_indexes=np.arange(len(options.seg_options.classes)))
    total_loss = dice_loss + focal_loss
    metrics = [sm.metrics.IOUScore(threshold=0.5, class_indexes=np.arange(len(options.seg_options.classes))), 
               sm.metrics.FScore(threshold=0.5, class_indexes=np.arange(len(options.seg_options.classes)))]
    optim = keras.optimizers.Adam(lr=options.seg_options.training_params.lr)

    weights_path = os.path.join(options.base_path, options.seg_options.training_params.model_weights_path)

    if weights_path is not None and os.path.isfile(weights_path):
        model.load_weights(weights_path)

    model.compile(optimizer=optim, loss=total_loss, metrics=metrics)
    return model

def getDepthEstimationModel(options: FoodRecognitionOptions):
    # Load depth estimation model
    print("Loading depth estimation model ...")

    custom_losses = Losses()
    objs = {'ProjectionLayer': ProjectionLayer,
            'ReflectionPadding2D': ReflectionPadding2D,
            'InverseDepthNormalization': InverseDepthNormalization,
            'AugmentationLayer': AugmentationLayer,
            'compute_source_loss': custom_losses.compute_source_loss}
    
    model_path = os.path.join(options.depth_options.training_params.model_path_json)
    model_weights_path = os.path.join(options.depth_options.training_params.model_weights_path)

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