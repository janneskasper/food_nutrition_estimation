import os
import json
from src.density_db import DensityDatabase
from src.ellipse_detection.ellipse_detector import EllipseDetector
from src.point_cloud_utils import sor_filter, pca_plane_estimation, align_plane_with_axis, pc_to_volume
from src.utils import get_cloud, _create_intrinsics_matrix
import numpy as np
import keras.backend as K
import cv2
import pandas as pd


class FoodNutritionCalculator:

    def __init__(self, nutrition_db_path: str, density_db_path: str, classes: list) -> None:
        assert os.path.isfile(nutrition_db_path), "Did not find DB nutrition file"
        assert os.path.isfile(density_db_path), "Did not find DB density file"

        self.density_db = DensityDatabase(density_db_path)

        with open(nutrition_db_path, "r") as f:
            print("[*] Loading nutrition database ...")
            self.nutrition_db: dict = json.load(f)

        self.classes = classes

    def calculateNutrition(self, volumes: dict):
        """ Uses predicted volumes, a nutrition database and a density database to calculate nutrition scores per food category

        Args:
            volumes (dict): Volme per food type (key: string food, value: float volume)

        Returns:
            dict: Nutrition score in json format
        """
        nutrition_scores = {}
        for k,v in volumes.items():
            if k in self.nutrition_db.keys():
                entry = self.nutrition_db[k]
                ref_weight = entry["reference_weight"]
                density = self.density_db.query(k)[1] # 0 is the name in db and 1 is the density
                weight = v * density
                print(f"[*] Weight of {k} is {weight} g, reference weight is {ref_weight} g, density is {density} g/cm3")
                nut_scores: dict = entry["scores"]
                nutrition_scores[k] = {}
                nutrition_scores[k]["predicted_weight"] = weight
                nutrition_scores[k]["predicted_volume"] = v
                nutrition_scores[k]["predicted_density"] = density
                nutrition_scores[k]["nutritional_values"] = {}
                for k_n, v_n in nut_scores.items():
                    nutrition_scores[k]["nutritional_values"][k_n] = v_n * (weight / ref_weight)
                
        return nutrition_scores

    def calculateScaling(self, input_img: np.ndarray,
                            disparity_map,
                            point_cloud,
                            plate_diameter_prior=0.3, 
                            gt_depth_scale=1.0):
        # Find ellipse parameterss (cx, cy, a, b, theta) that 
        # describe the plate contour
        ellipse_scale = 2
        ellipse_detector = EllipseDetector(input_shape=input_img.shape)
        ellipse_params = ellipse_detector.detect(input_img)
        ellipse_params_scaled = tuple([x / ellipse_scale for x in ellipse_params[:-1]] + [ellipse_params[-1]])

        # Scale depth map
        if point_cloud is not None and (any(x != 0 for x in ellipse_params_scaled) and plate_diameter_prior != 0):
            print('[*] Ellipse parameters:', ellipse_params_scaled)
            # Find the scaling factor to match prior 
            # and measured plate diameters
            plate_point_1 = [int(ellipse_params_scaled[2] 
                             * np.sin(ellipse_params_scaled[4]) 
                             + ellipse_params_scaled[1]), 
                             int(ellipse_params_scaled[2] 
                             * np.cos(ellipse_params_scaled[4]) 
                             + ellipse_params_scaled[0])]
            plate_point_2 = [int(-ellipse_params_scaled[2] 
                             * np.sin(ellipse_params_scaled[4]) 
                             + ellipse_params_scaled[1]),
                             int(-ellipse_params_scaled[2] 
                             * np.cos(ellipse_params_scaled[4]) 
                             + ellipse_params_scaled[0])]
            plate_point_1_3d = point_cloud[0, plate_point_1[0], 
                                           plate_point_1[1], :]
            plate_point_2_3d = point_cloud[0, plate_point_2[0], 
                                           plate_point_2[1], :]
            plate_diameter = np.linalg.norm(plate_point_1_3d 
                                            - plate_point_2_3d)
            scaling = plate_diameter_prior / plate_diameter
        else:
            # Use the median ground truth depth scaling when not using
            # the plate contour
            print('[*] No ellipse found. Scaling with expected median depth.')
            predicted_median_depth = np.median(1 / disparity_map)
            scaling = gt_depth_scale / predicted_median_depth
        print('[*] Scaling factor:', scaling)
        return scaling

    def calculateVolume(self, input_img: np.ndarray, 
                        segmentation_prediction: np.ndarray, 
                        disparity_map: np.ndarray, 
                        fov=70, 
                        relaxation_param=1.0,
                        plate_diameter_prior=0.3, 
                        gt_depth_scale=1.0):
        """ Calculates the volume for the given input. 

        Args:
            input_img (np.ndarray): The image used to generate the segmentations and depth
            segmentation_prediction (np.ndarray): Segmentation masks predicted from input image of shape (W,H,#classes+1)
            disparity_map (np.ndarray): Depth map predicted from input image of shape (W,H,1)
            fov (int, optional): FOV angle for intrinsic matrix calculation. Defaults to 70.
            plate_diameter_prior (float, optional): Diameter that can be used for scaling the objects. Defaults to 0.3.
            gt_depth_scale (float, optional): Ground truth for depth scaling. Defaults to 1.0.
            relaxation_param (float, optional): Relaxation parameter for plate diameter. Defaults to 1.0.
        Returns:
            dict: Dictionary from class name to calculated volume
        """
        assert segmentation_prediction.shape[-1] == len(self.classes) + 1, f"Classes don't match the segmentation prediction {len(self.classes)+1}:{segmentation_prediction.shape[2]}"

        tmp = np.zeros((disparity_map.shape[0], disparity_map.shape[1], segmentation_prediction.shape[2]))

        if segmentation_prediction.shape[-1] != disparity_map.shape[-1]:
            for i in range(segmentation_prediction.shape[-1]):
                tmp[...,i] = np.resize(segmentation_prediction[...,i], disparity_map.shape[:2])
            segmentation_prediction = tmp


        # Create intrinsics matrix
        intrinsics_mat = _create_intrinsics_matrix(disparity_map.shape[:2], fov)
        intrinsics_inv = np.linalg.inv(intrinsics_mat)

        depth = 1 / disparity_map
        # Convert depth map to point cloud
        depth_tensor = K.variable(np.expand_dims(depth, 0))
        intrinsics_inv_tensor = K.variable(np.expand_dims(intrinsics_inv, 0))
        point_cloud = K.eval(get_cloud(depth_tensor, intrinsics_inv_tensor))
        point_cloud_flat = np.reshape(point_cloud, (point_cloud.shape[1] * point_cloud.shape[2], 3))

        scaling = self.calculateScaling(input_img=input_img, 
                                        disparity_map=disparity_map, 
                                        point_cloud=point_cloud, 
                                        plate_diameter_prior=plate_diameter_prior, 
                                        gt_depth_scale=gt_depth_scale)
        depth = scaling * depth
        point_cloud = scaling * point_cloud
        point_cloud_flat = scaling * point_cloud_flat

        # Iterate over all predicted masks and estimate volumes
        estimated_volumes = {}
        for seg_class_index in range(segmentation_prediction.shape[-1]-1): # ignore the last one since its background
            # Apply mask to create object image and depth map
            object_mask = segmentation_prediction[...,seg_class_index] 
            object_depth = object_mask * depth
            # Get object/non-object points by filtering zero/non-zero 
            # depth pixels
            object_mask = (np.reshape(object_depth, (object_depth.shape[0] * object_depth.shape[1])) > 0)
            object_points = point_cloud_flat[object_mask, :]

            if not object_points.any():
                continue

            # Filter outlier points
            object_points_filtered, sor_mask = sor_filter(object_points, 2, 0.7)
            # Estimate base plane parameters
            plane_params = pca_plane_estimation(object_points_filtered)
            # Transform object to match z-axis with plane normal
            translation, rotation_matrix = align_plane_with_axis( plane_params, np.array([0, 0, 1]))
            object_points_transformed = np.dot(object_points_filtered + translation, rotation_matrix.T)

            # Adjust object on base plane
            height_sorted_indices = np.argsort(object_points_transformed[:,2])
            adjustment_index = height_sorted_indices[int(object_points_transformed.shape[0] * relaxation_param)]
            object_points_transformed[:,2] += np.abs(object_points_transformed[adjustment_index, 2])
             
            # Estimate volume for points above the plane
            volume_points = object_points_transformed[object_points_transformed[:,2] > 0]
            estimated_volume, _ = pc_to_volume(volume_points)
            estimated_volumes[self.classes[seg_class_index]] = estimated_volume * 10e6 # convert to cm3
            print(f"[*] Estimated volume for {self.classes[seg_class_index]}: {estimated_volume * 10e6} cm3")

        return estimated_volumes, scaling
