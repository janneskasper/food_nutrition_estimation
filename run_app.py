import os
from src.food_nutrition_calculator import FoodNutritionCalculator
from src.food_recognition_options import FoodRecognitionOptions
from src.models.model_factory import *
from src.utils import prettyPlotting, prettySegmentation
from flask import Flask, request, jsonify, make_response, abort
import base64
import requests
import io
from PIL import Image
import tensorflow as tf
import cv2

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class FoodNutritionApp:

    def __init__(self, options: FoodRecognitionOptions, visualize=False) -> None:

        self.seg_model = getSegmentationModel(config=options.seg_options.model_config, training=False, name_filter=None)
        self.calculator = FoodNutritionCalculator(options)
        self.visualize = options.seg_options.visualize
        self.depth_model = getDepthEstimationModel(config=options.depth_options.model_config)
        self.options = options
        self.depth_options = options.depth_options
        self.segment_options = options.seg_options
        self.graph = tf.compat.v1.get_default_graph()

        print("[*] Setup done!")

    def predictDepth(self, data: requests.Request):
        img, plate_diameter = self.__getData(data=data)
        img = cv2.resize(img, (224,128))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255
        img_batch = np.expand_dims(img, axis=0)

        with self.graph.as_default():
            inv_disp_map = self.depth_model.predict(img_batch, batch_size=1)[0][0,:,:,0]

        depth, disparity_map = self.__processDepthPrediction(np.array(inv_disp_map), None)

        if self.visualize: prettyPlotting([img, depth], (2,1), ['Input Image','Depth'], self.options.output_dir, 'Estimated Depth')

        # Return values
        return_vals = {
            "depth_img": self.__encodeImage(disparity_map)
        }
        return return_vals
    
    def predictSegmentation(self, data: requests.Request):
        img, plate_diameter = self.__getData(data=data)

        img_batch = np.expand_dims(img, axis=0)
        with self.graph.as_default():
            seg_masks = self.seg_model.predict(img_batch)[0]

        mask_onehot = self.__processSegmentation(seg_masks)

        pretty_mask = prettySegmentation(mask_onehot, self.segment_options.model_config.classes, self.segment_options.color_mapping)

        if self.visualize: prettyPlotting([img, pretty_mask], (2,1), ['Input Image','Combined Mask'], self.options.output_dir, 'Food Segmentation')

        return_vals = {
            "combined_mask": self.__encodeImage(pretty_mask)
        }
        return return_vals
    
    def predictNutrition(self, data: requests.Request):
        images, plate_diameter = self.__getData(data=data)
        image_batch = []
        image_depth_batch = []
        for img in images:
            img_resized = cv2.resize(img, (self.options.depth_options.model_config.input_size[1],
                                   self.options.depth_options.model_config.input_size[0]))
            img_correct_color = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB) / 255

            image_batch.append(img_resized)
            image_depth_batch.append(img_correct_color)

        image_batch = np.array(image_batch)
        image_depth_batch = np.array(image_depth_batch)

        # Predict segmentation and depth of batch
        with self.graph.as_default():
            seg_masks = self.seg_model.predict(image_batch)
            inv_disp_map = self.depth_model.predict(image_depth_batch, batch_size=1)[0]

        # For all images in batch, predict nutrition
        nut_scores_per_class = []
        for i in range(len(seg_masks)):
            disparity_map = (self.depth_options.min_disp + (self.depth_options.max_disp - self.depth_options.min_disp) * inv_disp_map[i,:,:,0])
            
            mask_onehot = self.__processSegmentation(seg_masks[i])

            with self.graph.as_default():
                volumes_per_class, scaling = self.calculator.calculateVolume(image_batch[i], mask_onehot, disparity_map, 
                                                                                fov=self.options.depth_options.fov, 
                                                                                gt_depth_scale=self.options.depth_options.gt_depth_scale,
                                                                                relaxation_param=self.options.seg_options.relax_param,
                                                                                plate_diameter_prior=plate_diameter, visualize=self.visualize)
            nut_scores_per_class.append(self.calculator.calculateNutrition(volumes_per_class))


        return nut_scores_per_class

    def __processSegmentation(self, output):
        output_raw = np.argmax(output, axis=2)
        masks_onehot = np.eye(len(self.segment_options.model_config.classes)+1)[output_raw]
        return masks_onehot

    def __processDepthPrediction(self, output, scaling):
        disparity_map = (self.depth_options.min_disp + (self.depth_options.max_disp - self.depth_options.min_disp) * output)
        depth = 1.0 / disparity_map
        if scaling is None:
            predicted_median_depth = np.median(depth)
            scaling = self.options.depth_options.gt_depth_scale / predicted_median_depth
        depth = scaling * depth
        return depth, disparity_map

    def __getData(self, data: requests.Request) -> tuple:
        try:
            content = json.loads(data.get_json())
            img_encoded = content['img']
            if isinstance(img_encoded, list):
                img = [self.__decodeImage(e) for e in img_encoded]
            else:
                img = [self.__decodeImage(img_encoded)]
        except Exception as e:
            print(e, flush=True)
            abort(406)
        try:
            plate_diameter = float(content['plate_diameter'])
        except Exception as e:
            plate_diameter = 0  
        
        return img, plate_diameter

    def __decodeImage(self, img):
        img_bytes = np.array(base64.b64decode(img.encode('utf-8')))
        img = Image.open(io.BytesIO(img_bytes))
        return np.array(img)

    def __encodeImage(self, img):  
        img = np.array(img)
        if len(img.shape) > 2 and img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if np.min(img) >= 0 and np.max(img) <= 1:
            img = (img*255)
        img = img.astype(np.uint8)
        img = Image.fromarray(img)
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format="PNG")
        img_byte_arr = img_byte_arr.getvalue()

        encoded_string = base64.b64encode(img_byte_arr).decode("utf-8")  
        return encoded_string


app = Flask("__name__")
food_nutrition_app: FoodNutritionApp = None


@app.route('/predict/depth', methods=["POST"])
def predictDepth():
    global food_nutrition_app
    res = food_nutrition_app.predictDepth(data=request)
    return make_response(jsonify(res), 200)


@app.route('/predict/nutrition', methods=["POST"])
def predictNutrition():
    global food_nutrition_app
    res = food_nutrition_app.predictNutrition(data=request)
    return make_response(jsonify(res), 200)


@app.route('/predict/segmentation', methods=["POST"])
def predictSegmentation():
    global food_nutrition_app
    res = food_nutrition_app.predictSegmentation(data=request)
    return make_response(jsonify(res), 200)


def run_app():
    global food_nutrition_app
    food_nutrition_app = FoodNutritionApp(FoodRecognitionOptions.createRunConfig())
    app.run(host="localhost", port=4333, debug=False)


if __name__ == "__main__":
    run_app()

