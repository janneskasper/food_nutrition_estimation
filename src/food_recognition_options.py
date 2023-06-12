import argparse
import os

TRAIN_ANNOTATIONS_PATH = "/home/jannes/Documents/MasterDelft/Q4/DeepLearning/datasets/food_rec/raw_data/public_training_set_release_2.0/annotations.json"
TRAIN_IMAGE_DIRECTORY = "/home/jannes/Documents/MasterDelft/Q4/DeepLearning/datasets/food_rec/raw_data/public_training_set_release_2.0/images/"

VAL_ANNOTATIONS_PATH = "/home/jannes/Documents/MasterDelft/Q4/DeepLearning/datasets/food_rec/raw_data/public_validation_set_2.0/annotations.json"
VAL_IMAGE_DIRECTORY = "/home/jannes/Documents/MasterDelft/Q4/DeepLearning/datasets/food_rec/raw_data/public_validation_set_2.0/images/"

TEST_IMAGE_DIRECTORY = "/home/jannes/Documents/MasterDelft/Q4/DeepLearning/datasets/food_rec/raw_data/public_test_release_2.0/images/"
WEIGHTS_PATH = "/home/jannes/Documents/MasterDelft/Q4/DeepLearning/tmp/train_20230602-204147/final_weights_only.hdf5"

class TrainingConfig:

    def __init__(self, args) -> None:
        self.log_dir = args.log_dir

        self.backbone = args.backbone
        self.input_size = args.input_size

        # self.batch_size = args.batch_size
        self.lr = args.lr
        self.lr_decay = args.lrd
        self.steps_epoch = args.steps_epoch
        self.epochs = args.epochs

        self.model_weights_path = None
        self.model_path_json = None

class SegmentationOptions:
    
    def __init__(self, args) -> None:
        self.training_params = TrainingConfig(args)
        
        self.relax_param = args.relax_param
        self.classes = [
                # 'water',
                # 'salad-leaf-salad-green',
                # 'bread-white',
                # 'tomato-raw',
                # 'butter', 
                # 'carrot-raw',
                # 'rice', 
                # 'egg', 
                'apple', 
                # 'jam', 
                # 'cucumber', 
                # 'banana', 
                # 'cheese', 
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

class DepthEstimationOptions:

    def __init__(self, args) -> None:
        self.training_params = TrainingConfig(args)

        self.min_depth = args.min_depth
        self.max_depth = args.max_depth
        self.min_disp = 1.0 / args.min_depth 
        self.max_disp = 1.0 / args.max_depth
        self.gt_depth_scale = args.gt_depth_scale
        self.fov = args.fov


class FoodRecognitionOptions:

    def __init__(self) -> None:
        parsing_strat = False

        args = self.__parse_args(parsing_strat) if parsing_strat else self.__parse_args(parsing_strat)[0]

        self.seg_options = SegmentationOptions(args)
        self.depth_options = DepthEstimationOptions(args)

        self.backbone = args.backbone
        self.batch_size = int(args.batch_size)
        self.input_size = args.input_size
        self.lr = float(args.lr)
        self.lrd = float(args.lrd)
        self.steps_epoch = int(args.steps_epoch)
        self.epochs = int(args.epochs)
        self.log_dir = args.log_dir

        self.train_ann_path = args.train_ann
        self.val_ann_path = args.val_ann
        self.train_img_path = args.train_img
        self.val_img_path = args.val_img
        self.test_img_path = TEST_IMAGE_DIRECTORY
        
        self.nut_db: str="data/nutrition_db.json"
        self.density_db: str="data/density_db.xlsx"
        
        self.base_path: str = os.getcwd()

    def __parse_args(self, full=True):
        parser = argparse.ArgumentParser()
        parser.add_argument("-b", "--batch_size", type=int, default=32, 
                            help="Defines the training batch size")
        parser.add_argument("-i", "--input_size", type=tuple, default=(224,224,3), 
                            help="Defines the input size")
        parser.add_argument("-lr", type=float, default=0.001, 
                            help="Defines the initial learning rate")
        parser.add_argument("-lrd", type=float, default=0.00001, 
                            help="Defines the learning rate decay")
        parser.add_argument("--backbone", type=str, default="resnet18", 
                            help="Defines the model backbone")
        parser.add_argument("-s", "--steps_epoch", type=int, default=200, 
                            help="Defines the training steps per epoch")
        parser.add_argument("-e", "--epochs", type=int, default=50, 
                            help="Defines the training number of epochs")
        parser.add_argument("--log_dir", type=str, default="./tmp", 
                            help="Defines where the checkpoints and ouputs are stored")
        parser.add_argument("--fov", type=float, default=70.0, 
                            help="Min prediction depth")
        parser.add_argument("--min_depth", type=float, default=0.01, 
                            help="Min prediction depth")
        parser.add_argument("--max_depth", type=float, default=10.0, 
                            help="Max prediction depth")
        parser.add_argument("--gt_depth_scale", type=float, default=0.35, 
                            help="Max prediction depth")
        parser.add_argument("--relax_param", type=float, default=0.1, 
                            help="Max prediction depth")
        parser.add_argument("-ta", "--train_ann", type=str, default=TRAIN_ANNOTATIONS_PATH,
                            help="Path to training annotations")
        parser.add_argument("-ti", "--train_img", type=str, default=TRAIN_IMAGE_DIRECTORY,
                            help="Path to training image dir")
        parser.add_argument("-va", "--val_ann", type=str, default=VAL_ANNOTATIONS_PATH,
                            help="Path to validation annotations")
        parser.add_argument("-vi", "--val_img", type=str, default=VAL_IMAGE_DIRECTORY,
                            help="Path to training image dir")

        if full:
            return parser.parse_args()
        return parser.parse_known_args()

    @staticmethod
    def create():
        f = FoodRecognitionOptions()
        f.depth_options.training_params.model_path_json = "model_files/monovideo_fine_tune_food_videos.json"
        f.depth_options.training_params.model_weights_path = "model_files/monovideo_fine_tune_food_videos.h5"

        # f.seg_options.training_params.model_weights_path =  "model_files/seg_model_e18.hdf5"
        
        # apples
        f.seg_options.training_params.model_weights_path = "logs/train_20230611-230608/final_model.hdf5"
        
        return f

