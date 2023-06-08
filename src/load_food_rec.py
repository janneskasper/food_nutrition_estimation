from collections import OrderedDict
from pycocotools.coco import COCO
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
import random
import plotly.graph_objects as go
import keras
import shutil
# from keras.preprocessing.image import ImageDataGenerator

# heavily based on the tutorial from Viraf Patrawala (March, 2020)
# https://github.com/virafpatrawala/COCO-Semantic-Segmentation/blob/master/COCOdataset_SemanticSegmentation_Demo.ipynb

# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show() 

class CocoDatasetGenerator:

    def __init__(self, annotations_path: str, 
                 img_dir: str, 
                 preprocessor=None,
                 filter_categories=None, 
                 data_size=(455,455), 
                 batch_size=12) -> None:
        """ Initialize CocoDatasetGenerator

        Args:
            annotations_path (str): Absolute path to the coco annotations file
            img_dir (str): Absolute path to the corresponding image directory
            filter_categories (list, optional): List of category names to use. Uses all if None. Defaults to None.
            data_size (tuple, optional): The size of the generator output images and masks. Defaults to (455,455).
        """
        assert os.path.isfile(annotations_path), f"Provided path for train annotations is invalid!"
        self.annotations_path = annotations_path
        self.img_dir = img_dir
        self.data_size = data_size

        self.preprocessor = preprocessor
        
        self.coco_obj = COCO(self.annotations_path)

        self.filterDataset(filter_categories)

        self.n_classes = len(self.categoryNames) 
        self.batch_size = batch_size

        
    def __len__(self):
        return len(self.imgs) // self.batch_size

    def getClassDistribution(self, normalize=True):
        no_images_per_category = {}
        total = 0
        for n, cat in enumerate(self.categories):
            imgIds = self.coco_obj.getImgIds(catIds=cat['id'])
            index = self.categoryNames.index(cat["name"])
            no_images_per_category[index] = len(imgIds)
            total += len(imgIds)
        if normalize:
            for k,v in no_images_per_category.items():
                no_images_per_category[k] = v / total
        return no_images_per_category

    def filterDataset(self, categories: list):  
        """ Filters the loaded dataset by the given list of category names.

        Args:
            categories (list): List of category names to use.
        """
        if categories is not None:
            catIds = self.coco_obj.getCatIds(catNms=categories)
        else:
            catIds = self.coco_obj.getCatIds()
        imgIds = []
        for catId in catIds:
            imgIds += self.coco_obj.catToImgs[catId] 
        self.imgs = self.coco_obj.loadImgs(imgIds)
        self.annotations = self.coco_obj.loadAnns(self.coco_obj.getAnnIds(catIds=catIds))
        self.categories = self.coco_obj.loadCats(catIds)
        self.categoryNames = [cat["name"] for cat in self.categories]

        self.annotations_per_img = {}
        for img in self.imgs: 
            annIds = self.coco_obj.getAnnIds(img['id'], catIds=catIds, iscrowd=None)
            anns = self.coco_obj.loadAnns(annIds)
            self.annotations_per_img[img["id"]] = anns
                
        random.shuffle(self.imgs)

    def getCategoryName(self, category_id):
        """ Returns the category name for a given id.

        Args:
            classId (int): Category ID to find the corresponding name for

        Returns:
            str: Category name or None if not found
        """
        for cat in self.categories:
            if cat['id']==category_id:
                return cat['name']
        return None

    def generateMask(self, img, anns: list):
        """ Generate the segmentation mask for a given coco image object.

        Args:
            img (dict): Coco image object to generate the segmentation mask for (Mask size: (img["height"], img["width"], 1))

        Returns:
            tuple: Segmentation mask image, Binary Segmentation mask stack of depth (# classes) 
        """
        masks = np.zeros((self.data_size[0], self.data_size[1], len(self.categoryNames)+1), np.float64)
        background = np.ones((self.data_size[0], self.data_size[1]), dtype=np.float32)

        checked_classes = np.zeros(len(self.categoryNames))
        created_maskes = {}
        for i, ann in enumerate(anns):
            index = self.categoryNames.index(self.coco_obj.loadCats([ann["category_id"]])[0]['name'])
            mask = self.coco_obj.annToMask(ann)
            if not mask.shape == (self.data_size[0], self.data_size[1], 1):
                mask = cv2.resize(mask, (self.data_size[0], self.data_size[1]), interpolation=cv2.INTER_NEAREST)
            if checked_classes[index]:
                created_maskes[index] = np.logical_or(created_maskes[index], mask)
            else:
                checked_classes[index] = 1
                created_maskes[index] = mask

        for ind, mas in created_maskes.items():
            background = np.maximum(background - mas, 0.0)
            masks[...,ind] = mas

        masks[...,-1] = background.astype(np.float64)
        
        return masks

    def loadImage(self, image_obj: dict):
        """ Loads the real image for a given coco image object.

        Args:
            image_obj (dict): Coco image object to load the image for 

        Returns:
            array: The loaded color image
        """
        img = cv2.imread(os.path.join(self.img_dir, image_obj['file_name']))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if not img.shape == self.data_size: 
            img = cv2.resize(img, self.data_size[:2])
        if (len(img.shape)==3 and img.shape[2]==3):
            return img
        else:
            stacked_img = np.stack((img,)*3, axis=-1)
            return stacked_img

    def getGenerator(self):
        """ Creates the dataset generator with given batch size and possible filtering,

        Args:
            categories (list, optional): List of category names to use. Uses all if None. Defaults to None.
            batch_size (int, optional): Defines the size of the return batch. Defaults to 4.

        Yields:
            tuple: Batch of (images, masks) with a first dimension of "batch_size"
        """
        iteration_cnt = 0
        iteration_max = len(self.imgs)

        while True:
            img_batch = np.zeros((self.batch_size, self.data_size[0], self.data_size[1], 3)).astype(np.float64)
            mask_batch = np.zeros((self.batch_size, self.data_size[0], self.data_size[1], self.n_classes+1)).astype(np.float64)
            for i in range(iteration_cnt, iteration_cnt+self.batch_size):
                img_obj = self.imgs[i]
                img = self.loadImage(img_obj)
                mask = self.generateMask(img, self.annotations_per_img[img_obj["id"]])
                if self.preprocessor:
                    sample = self.preprocessor(image=img, mask=mask)
                    img, mask = sample['image'], sample['mask']

                img_batch[i-iteration_cnt] = img
                mask_batch[i-iteration_cnt] = mask

            iteration_cnt += self.batch_size
            if iteration_cnt + self.batch_size >= iteration_max:
                iteration_cnt = 0
                random.shuffle(self.imgs)

            yield img_batch, mask_batch
    
    def __getitem__(self, i):
        iteration_max = len(self.imgs)
        iteration_cnt = i

        if iteration_cnt + self.batch_size >= iteration_max:
            iteration_cnt = 0
            random.shuffle(self.imgs)

        img_batch = np.zeros((self.batch_size, self.data_size[0], self.data_size[1], 3)).astype(np.float64)
        mask_batch = np.zeros((self.batch_size, self.data_size[0], self.data_size[1], self.n_classes+1)).astype(np.float64)

        for i in range(iteration_cnt, iteration_cnt+self.batch_size):
            img_obj = self.imgs[i]
            img = self.loadImage(img_obj)
            mask = self.generateMask(img, self.annotations_per_img[img_obj["id"]])
            if self.preprocessor:
                sample = self.preprocessor(image=img, mask=mask)
                img, mask = sample['image'], sample['mask']

            img_batch[i-iteration_cnt] = img
            mask_batch[i-iteration_cnt] = mask

        iteration_cnt += self.batch_size

        return img_batch, mask_batch

    def visualizeGenerator(self, gen):
        """ Visualizes 4 examples of a given generator. 

        Args:
            gen (generator): Generator to call "next()" on to get data
        """
        img, mask = next(gen)
        fig = plt.figure(figsize=(20, 10))
        outerGrid = gridspec.GridSpec(1, 2, wspace=0.1, hspace=0.1)
        
        for i in range(2):
            innerGrid = gridspec.GridSpecFromSubplotSpec(2, 2,
                            subplot_spec=outerGrid[i], wspace=0.05, hspace=0.05)

            for j in range(4):
                ax = plt.Subplot(fig, innerGrid[j])
                if(i==1):
                    ax.imshow(img[j])
                else:
                    ax.imshow(mask[j][:,:,0])
                    
                ax.axis('off')
                fig.add_subplot(ax)        
        plt.show()

    def extractImages(self, in_path: str, out_path: str):
        assert os.path.isdir(in_path), f"Given input path should be a directory of images"
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        copy_cnt = 0
        for img in self.imgs:
            img_id = img['id']
            in_img_name = img["file_name"]
            cats = []
            img_name = ""
            for ann in self.annotations_per_img[img_id]:
                cat_name = self.coco_obj.loadCats([ann["category_id"]])[0]['name']
                if not cat_name in cats:
                    cats.append(cat_name)
                    img_name += f"{cat_name}_"
            cnt = 0
            outname = img_name + f"{cnt}"
            out_img_path = os.path.join(out_path, f"{outname}.jpg")
            while os.path.exists(out_img_path):
                cnt += 1
                outname = img_name + f"{cnt}"
                out_img_path = os.path.join(out_path, f"{outname}.jpg")

            shutil.copy2(os.path.join(in_path, in_img_name), out_img_path)
            copy_cnt += 1

        print(f"Copied {copy_cnt} images to {out_path}...")

    def visualizeDatasplit(self, num_relevant=30):
        no_images_per_category = {}
        categories = self.coco_obj.loadCats(self.coco_obj.getCatIds())
        for n, i in enumerate(self.coco_obj.getCatIds()):
            imgIds = self.coco_obj.getImgIds(catIds=i)
            label = categories[n]["name"]
            no_images_per_category[label] = len(imgIds)

        no_images_per = OrderedDict(sorted(no_images_per_category.items(), key=lambda x: -1*x[1]))
        i = 0
        no_images_per_category = {}
        categorie_names = []
        for k, v in no_images_per.items():
            print(k, v)
            categorie_names.append(k)
            no_images_per_category[k] = v
            i += 1
            if i > num_relevant:
                break
        print(categorie_names)
        fig = go.Figure([go.Bar(x=list(no_images_per_category.keys()), y=list(no_images_per_category.values()))])
        fig.update_layout(
            title="No of Image per class", )
        fig.show()
        fig = go.Figure(data=[go.Pie(labels=list(no_images_per_category.keys()), values=list(no_images_per_category.values()),
                                    hole=.3, textposition='inside', )], )
        fig.update_layout(
            title="No of Image per class ( In pie )", )
        fig.show()

class CocoDatasetLoader(keras.utils.Sequence):
    
    def __init__(self, dataset: CocoDatasetGenerator) -> None:
        super().__init__()
        self.dataset = dataset
        self.dataset_gen = dataset.getGenerator()

    def __getitem__(self, index):

        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

if __name__ == "__main__":
    TRAIN_ANNOTATIONS_PATH = "/home/jannes/Documents/MasterDelft/Q4/DeepLearning/datasets/food_rec/raw_data/public_training_set_release_2.0/annotations.json"
    TRAIN_IMAGE_DIRECTORY = "/home/jannes/Documents/MasterDelft/Q4/DeepLearning/datasets/food_rec/raw_data/public_training_set_release_2.0/images/"

    VAL_ANNOTATIONS_PATH = "/home/jannes/Documents/MasterDelft/Q4/DeepLearning/datasets/food_rec/raw_data/public_validation_set_2.0/annotations.json"
    VAL_IMAGE_DIRECTORY = "/home/jannes/Documents/MasterDelft/Q4/DeepLearning/datasets/food_rec/raw_data/public_validation_set_2.0/images/"
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
            #   'wine-red', 
            #   'banana', 
            #   'cheese', 
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
    base = os.getcwd()
    m = CocoDatasetGenerator(VAL_ANNOTATIONS_PATH, VAL_IMAGE_DIRECTORY, filter_categories=CLASSES, batch_size=32)
    m.extractImages(os.path.join(base,"datasets/food_rec/raw_data/public_validation_set_2.0/images"),
                    os.path.join(base ,"datasets/filtered"))