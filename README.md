# Food Nutrition Estimation
Start the app:
- Call: **python food_nutrition_app.py** to start the server
- Call: **python test_client.py** to start a test
- Call: **python src/train_food_rec.py** to train your custom model (first configure the training process in **[food_recognition_options.py](./src/food_recognition_options.py)**)

## Get the network models
Download the following files to **[model_dir](./src/models/files)**.
- https://drive.google.com/file/d/1o0nzFNoW74EaW98XIQkHxEyGGB4UG6WL/view?usp=sharing -> Weights depth model
- https://drive.google.com/file/d/1L7tkIX8V1-TOL2xiNHbYEW_kCrQGHpBJ/view?usp=sharing -> Architecture depth model
- https://drive.google.com/file/d/1o_LvDI9ISedsZSXaCBBcuict0eqWP7Od/view?usp=sharing -> Weights segmentation model

## Configure the application and training processes
The file **[food_recognition_options.py](./src/food_recognition_options.py)** contains all the configurations for training the networks and starting the server.


### Training
```
python train.py -b 32  -ta ..\food_volume_estimation\datasets\raw_data\public_training_set_release_2.0\annotations.json -ti ..\food_volume_estimation\datasets\raw_data\public_training_set_release_2.0\images\ -va ..\food_volume_estimation\datasets\raw_data\public_validation_set_2.0\annotations.json -vi ..\food_volume_estimation\datasets\raw_data\public_validation_set_2.0\images\ --backbone="resnet18" --log_dir=logs --weights .\model_files\apple_model.hdf5
```

## Testing

### Server
If you want to get images of the results you can do the following command:
```
python .\run_app.py --weights <path_to_model_weights> --visualize --output_dir <output_dir>
```

Otherwise you can just run:
```
python .\run_app.py --weights <path_to_model_weights>
```

### Client
You can test using the following command once the server is running:
```
python .\test_client.py <file/directory pointing to images>
```