import cv2
import requests
import json
import sys
import os
import base64
from PIL import Image
import io
import numpy as np
import os

def encode_image(img_path, size=(224,224)):
    if isinstance(img_path, str):
        im = Image.open(img_path)
        im = im.resize(size)
    elif isinstance(img_path, np.ndarray):
        im = Image.fromarray(img_path)
        im = im.resize(size)

    img_byte_arr = io.BytesIO()
    im.save(img_byte_arr, format="PNG")
    img_byte_arr = img_byte_arr.getvalue()
    encoded_string = base64.b64encode(img_byte_arr).decode("utf-8") 

    return encoded_string   


def test_run(img_path, size=(224,224), type="seg"):
    if isinstance(img_path, str):
        encoded_string = [encode_image(img_path, size=size)]
    elif isinstance(img_path, list):
        encoded_string = [encode_image(e, size=size) for e in img_path]
        
    if type == "seg":
        data = requests.post("http://localhost:4333/predict/segmentation", json=json.dumps({"img": encoded_string, "plate_diameter": 0.422}),)
        if data.status_code == 200:
            try:
                content = json.loads(data.content)
                img_encoded = content['combined_mask']
                img_bytes = np.array(base64.b64decode(img_encoded.encode('utf-8')))
                img = Image.open(io.BytesIO(img_bytes))
                img = np.array(img)
            except Exception as e:
                print(e, flush=True)
                return
            cv2.imshow("RGB", img)
            cv2.waitKey(0)
    elif type == "depth":
        data = requests.post("http://localhost:4333/predict/depth", json=json.dumps({"img": encoded_string, "plate_diameter": 0.422}),)
        if data.status_code == 200:
            try:
                content = json.loads(data.content)
                img_encoded = content['depth_img']
                img_bytes = np.array(base64.b64decode(img_encoded.encode('utf-8')))
                img = Image.open(io.BytesIO(img_bytes))
                img = np.array(img)
            except Exception as e:
                print(e, flush=True)
                return
            cv2.imshow("RGB", img)
            cv2.waitKey(0)
    elif type == "nut":
        data = requests.post("http://localhost:4333/predict/nutrition", json=json.dumps({"img": encoded_string, "plate_diameter": 0.3}),)
        if data.status_code == 200:
            try:
                content = json.loads(data.content)
            except Exception as e:
                print(e, flush=True)
                return
            return content
        
def getAverageNutrition(content):
    if content and isinstance(content, list):
        avg_nutrition = {}
        for food_result in content:
            for food_type in food_result.keys():
                if food_type not in avg_nutrition.keys():
                    avg_nutrition[food_type] = {}
                
                # Sum up volume and weight
                if "volume" not in avg_nutrition[food_type].keys():
                    avg_nutrition[food_type]["volume"] = 0
                if "weight" not in avg_nutrition[food_type].keys():
                    avg_nutrition[food_type]["weight"] = 0
                
                avg_nutrition[food_type]["volume"] += food_result[food_type]["predicted_volume"]
                avg_nutrition[food_type]["weight"] += food_result[food_type]["predicted_weight"]

                # Sum up nutritional values
                for k,v in food_result[food_type]["nutritional_values"].items():
                    if k not in avg_nutrition[food_type].keys():
                        avg_nutrition[food_type][k] = 0
                    avg_nutrition[food_type][k] += v
        
        for k,v in avg_nutrition.items():
            for k1,v1 in v.items():
                avg_nutrition[k][k1] = v1 / len(content)

        return avg_nutrition
    
def getImagesFromVideo(video_path, rate=1):
    """
    Retrieve a list of Image from a video
    
    input: video_path (str)
    output: list of Image
    """
    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()
    counter = 0
    images = []
    while success:
        if counter % rate == 0:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)
        success,image = vidcap.read()
        counter += 1
    return images


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 1:
        if os.path.isdir(args[0]):
            test_img = [f"{args[0]}/{path}" for path in os.listdir(args[0])]
        else:
            test_img = args[0]
    else:
        # test_img = f"{os.getcwd()}/../datasets/filtered/apple_0.jpg"
        # test_img = f"{os.getcwd()}/../datasets/filtered/water_banana_3.jpg"
        # test_img = f"{os.getcwd()}/../datasets/filtered/water_100.jpg"
        # test_img = f"{os.getcwd()}/../datasets/filtered/tomato-raw_salad-leaf-salad-green_0.jpg"
        # test_img = f"{os.getcwd()}/../datasets/filtered/salad-leaf-salad-green_25.jpg"
        # test_img = f"{os.getcwd()}/../datasets/filtered/rice_example.jpg"
        # test_img = ["data/test_images/plate20cm/red/"+ path for path in os.listdir("data/test_images/plate20cm/red")]
        test_img = f"{os.getcwd()}/../datasets/filtered/carrot-raw_17.jpg"
        # test_img = "data/test_images/plate25cm/applered2.jpg"
        # test_img = getImagesFromVideo("data/test_images/video/60fpsfhd.mp4", rate=5)

    
    # test_run(test_img, type="seg")
    # test_run(test_img, type="depth")
    content = test_run(test_img, type="nut")
    
    print(getAverageNutrition(content))
