import argparse
import os
import json


ALL_CLASSES = ['water',
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
               'bread-wholemeal', 
               'coffee-with-caffeine', 
               'mixed-vegetables', 
               'wine-red', 
               'potatoes-steamed', 
               'bell-pepper-red-raw', 
               'hard-cheese', 
               'espresso-with-caffeine', 
               'tea', 
               'bread-whole-wheat', 
               'mixed-salad-chopped-without-sauce', 
               'avocado', 
               'white-coffee-with-caffeine', 
               'tomato-sauce', 
               'wine-white', 
               'broccoli', 
               'strawberries', 
               'pasta-spaghetti'
                ]
COLOR_MAPPING = {
                'water': (0, 0, 255),
                'salad-leaf-salad-green': (0, 255, 0),
                'bread-white': (255, 255, 255),
                'tomato-raw': (255, 0, 0),
                'butter': (255, 255, 0),
                'carrot-raw': (255, 165, 0),
                'rice': (255, 192, 203),
                'egg': (255, 0, 255),
                'apple': (0, 255, 0),
                'jam': (128, 0, 128),
                'cucumber': (0, 255, 255),
                'banana': (255, 215, 0),
                'cheese': (255, 127, 80),
                'bread-wholemeal': (210, 180, 140),
                'coffee-with-caffeine': (0, 0, 0),
                'mixed-vegetables': (0, 128, 128),
                'wine-red': (128, 0, 0),
                'potatoes-steamed': (128, 128, 128),
                'bell-pepper-red-raw': (255, 0, 255),
                'hard-cheese': (165, 42, 42),
                'espresso-with-caffeine': (75, 0, 130),
                'tea': (0, 0, 128),
                'bread-whole-wheat': (218, 112, 214),
                'mixed-salad-chopped-without-sauce': (221, 160, 221),
                'avocado': (64, 224, 208),
                'white-coffee-with-caffeine': (240, 230, 140),
                'tomato-sauce': (255, 255, 0),
                'wine-white': (192, 192, 192),
                'broccoli': (0, 128, 0),
                'strawberries': (255, 0, 0),
                'pasta-spaghetti': (255, 165, 0)
                }


class TrainingConfig:

    def __init__(self) -> None:
        args = self.__parse_args()
        self.model_options = ModelConfig(args)

        self.batch_size = int(args.batch_size)
        self.lr = float(args.lr)
        self.lrd = float(args.lrd)
        self.steps_epoch = int(args.steps_epoch)
        self.epochs = int(args.epochs)
        self.workers = int(args.workers)
        self.class_weights = None

        self.log_dir = args.log_dir
        self.train_ann_path = args.train_ann
        self.val_ann_path = args.val_ann
        self.train_img_path = args.train_img
        self.val_img_path = args.val_img
        self.test_img_path = "./"

    def saveToJson(self, path):
        json_dict = json.dumps(self, default=lambda o: o.__dict__,  sort_keys=True, indent=4)
        
        with open(path, "w") as f:
            f.write(json_dict)

    def __parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-w", "--workers", type=int, 
                            default=8, 
                            help="Defines the number of training workers")
        parser.add_argument("-b", "--batch_size", type=int, 
                            default=32, 
                            help="Defines the training batch size")
        parser.add_argument("-i", "--input_size", type=tuple, 
                            default=(224,224,3), 
                            help="Defines the input size")
        parser.add_argument("-lr", type=float, 
                            default=0.001, 
                            help="Defines the initial learning rate")
        parser.add_argument("-lrd", type=float, 
                            default=0.000001, 
                            help="Defines the learning rate decay")
        parser.add_argument("--backbone", type=str, 
                            default="resnet18", 
                            help="Defines the model backbone")
        parser.add_argument("--model", type=str, 
                            default="unet", 
                            help="Defines the model backbone")
        parser.add_argument("-s", "--steps_epoch", type=int, 
                            default=100, 
                            help="Defines the training steps per epoch")
        parser.add_argument("-e", "--epochs", type=int, 
                            default=50, 
                            help="Defines the training number of epochs")
        parser.add_argument("--log_dir", type=str, default="./tmp", 
                            help="Defines where the checkpoints and ouputs are stored")
        parser.add_argument("-ta", "--train_ann", type=str, 
                            default=os.path.join(os.getcwd(), "../datasets/food_rec/raw_data/public_training_set_release_2.0/annotations.json"),
                            help="Path to training annotations")
        parser.add_argument("-ti", "--train_img", type=str, 
                            default=os.path.join(os.getcwd(), "../datasets/food_rec/raw_data/public_training_set_release_2.0/images"),
                            help="Path to training image dir")
        parser.add_argument("-va", "--val_ann", type=str, 
                            default=os.path.join(os.getcwd(), "../datasets/food_rec/raw_data/public_validation_set_2.0/annotations.json"),
                            help="Path to validation annotations")
        parser.add_argument("-vi", "--val_img", type=str, 
                            default=os.path.join(os.getcwd(), "../datasets/food_rec/raw_data/public_validation_set_2.0/images"),
                            help="Path to training image dir")
        parser.add_argument("--weights", type=str, default="model_files/seg_model_e18.hdf5",
                            help="Path to segmentation model weights")

        return parser.parse_args()
    
    @staticmethod
    def createTrainConfig():
        opt = TrainingConfig()

        opt.epochs = 1
        opt.lr = 0.001
        opt.lrd = 0.0001

        opt.model_options.model_weights_path = None

        opt.model_options.classes = [
                                    'bread-white',
                                    'apple', 
                                    'carrot-raw',
                                    ]
        
        opt.model_options.model_backbone = "resnet18"
        opt.model_options.model = "unet"

        return opt



class FoodRecognitionOptions:

    def __init__(self) -> None:

        args = self.__parse_args()

        self.seg_options = SegmentationOptions(args)
        self.depth_options = DepthEstimationOptions(args)

        self.output_dir = args.output_dir
        
        self.nut_db: str="data/nutrition_db.json"
        self.density_db: str="data/density_db.xlsx"
        
        self.base_path: str = os.getcwd()

    def __parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--fov", type=float, default=70.0, 
                            help="Depth estimation FOV")
        parser.add_argument("--min_depth", type=float, default=0.01, 
                            help="Min prediction depth")
        parser.add_argument("--max_depth", type=float, default=10.0, 
                            help="Max prediction depth")
        parser.add_argument("--gt_depth_scale", type=float, default=0.65, 
                            help="Ground truth for depth scaling")
        parser.add_argument("--relax_param", type=float, default=0.1, 
                            help="Relaxation parameter")
        parser.add_argument("-i", "--input_size", type=tuple, default=(128,224,3), 
                            help="Defines the input size")
        parser.add_argument("--backbone", type=str, default="resnet18", 
                            help="Defines the model backbone")
        parser.add_argument("--weights", type=str, default="model_files/seg_model_e18.hdf5",
                            help="Path to segmentation model weights")
        parser.add_argument("--visualize", action='store_true',
                            help="Visualize intermediate results")
        parser.add_argument("--output_dir", type=str, default="output",
                            help="Path to directory to store output images")
        return parser.parse_args()

    @staticmethod
    def createRunConfig():
        opt = FoodRecognitionOptions()
        opt.depth_options.model_config.model_path_json = "model_files/monovideo_fine_tune_food_videos.json"
        opt.depth_options.model_config.model_weights_path = "model_files/monovideo_fine_tune_food_videos.h5"

        opt.seg_options.model_config.model_weights_path = opt.seg_options.weights
        
        opt.seg_options.model_config.model_weights_path = "tmp/train_20230615-152539/final_model.hdf5"

        opt.seg_options.model_config.classes = [
                                                "carrot-raw",
                                                "apple",
                                                "bread-white"
                                                ]
        return opt


class ModelConfig:

    def __init__(self, args) -> None:
        self.model_backbone = args.backbone
        self.model = args.model
        self.model_weights_path = None
        self.model_path_json = None
        self.input_size = args.input_size
        self.classes = []

    def toJson(self):
        return json.dumps(self, default=lambda o: o.__dict__,  sort_keys=True, indent=4)


class SegmentationOptions:
    
    def __init__(self, args) -> None:        
        self.model_config = ModelConfig(args)

        self.relax_param = args.relax_param
        self.weights = args.weights
        self.visualize = args.visualize

        self.color_mapping = COLOR_MAPPING


class DepthEstimationOptions:

    def __init__(self, args) -> None:
        self.model_config = ModelConfig(args)

        self.min_depth = args.min_depth
        self.max_depth = args.max_depth
        self.min_disp = 1.0 / args.min_depth 
        self.max_disp = 1.0 / args.max_depth
        self.gt_depth_scale = args.gt_depth_scale
        self.fov = args.fov
