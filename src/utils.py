
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Geometry utilities for projecting frames based on depth and motion.

Modified from Spatial Transformer Networks:
https://github.com/tensorflow/models/blob/master/transformer/spatial_transformer.py

Taken from:
    https://github.com/tensorflow/models/tree/master/research/struct2depth
Modified by Alexander Graikos
"""

import os
from absl import logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time

def prettySegmentation(one_hot_mask: np.ndarray, classes: list, color_scheme: dict):
    """ Combines one hot encoded segmentation mask based on a color scheme.

    Args:
        one_hot_mask (np.ndarray): One hot encoded masks
        color_scheme (dict): Color scheme with class names as keys and colors as value

    Returns:
        np.ndarray: Combined colored mask
    """
    pretty_mask = np.zeros((one_hot_mask.shape[0], one_hot_mask.shape[1], 3))
    for i, class_i in enumerate(classes):
        m = one_hot_mask[...,i] == 1.0
        if len(m) == 0:
            continue
        pretty_mask[m] = color_scheme[class_i]
    return pretty_mask

def prettyPlotting(imgs, tiling, titles, dir="out", suptitle=None):
    """Plot images in a pretty grid.

    Inputs:
        imgs: List of images to plot.
        tiling: Subplot tiling tuple (rows,cols).
        titles: List of subplot titles.
        suptitle: Suptitle above all plots.
    """
    if not os.path.isdir(dir):
        os.makedirs(dir)

    n_plots = len(imgs)
    rows = str(tiling[0])
    cols = str(tiling[1])
    fig = plt.figure()
    for r in range(tiling[0] * tiling[1]):
        plt.subplot(rows + cols + str(r + 1))
        plt.title(titles[r])
        plt.imshow(imgs[r])

    if suptitle is not None:
        fig.suptitle(suptitle)
    # plt.show()
    plt.savefig("{}/{}.png".format(dir,time.strftime("%Y%m%d-%H%M%S")))
    plt.close()

def inverse_warp(img, depth, pose_vector, intrinsic_mat, intrinsic_mat_inv):
    """Inverse warp a source image to the target image plane.

    Args:
        img: The source image (to sample pixels from) -- [B, H, W, 3].
        depth: Depth map of the target image -- [B, H, W].
        pose_vec: Pose transform vector -- [B, 6].
        intrinsic_mat: Camera intrinsic matrix -- [3, 3].
        intrinsic_mat_inv: Inverse of the intrinsic matrix -- [3, 3].
    Returns:
        Projected source image
    """
    dims = tf.shape(img)
    batch_size, img_height, img_width = dims[0], dims[1], dims[2]

    # Get transformation matrix from pose vector
    pose_mat = _egomotion_vec2mat(pose_vector, batch_size)
    # Expand intrinsics to batch size
    intrinsic_mat = tf.tile(tf.expand_dims(intrinsic_mat, axis=0),
                            [batch_size, 1, 1])
    intrinsic_mat_inv = tf.tile(tf.expand_dims(intrinsic_mat_inv, axis=0),
                                [batch_size, 1, 1])

    depth = tf.reshape(depth, [batch_size, 1, img_height * img_width])
    grid = _meshgrid_abs(img_height, img_width)
    grid = tf.tile(tf.expand_dims(grid, 0), [batch_size, 1, 1])
    cam_coords = _pixel2cam(depth, grid, intrinsic_mat_inv)
    ones = tf.ones([batch_size, 1, img_height * img_width])
    cam_coords_hom = tf.concat([cam_coords, ones], axis=1)

    # Get projection matrix for target camera frame to source pixel frame
    hom_filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
    hom_filler = tf.tile(hom_filler, [batch_size, 1, 1])
    intrinsic_mat_hom = tf.concat([intrinsic_mat, tf.zeros([batch_size, 3, 1])],
                                  axis=2)
    intrinsic_mat_hom = tf.concat([intrinsic_mat_hom, hom_filler], axis=1)
    proj_target_cam_to_source_pixel = tf.matmul(intrinsic_mat_hom, pose_mat)
    source_pixel_coords = _cam2pixel(cam_coords_hom,
                                     proj_target_cam_to_source_pixel)
    source_pixel_coords = tf.reshape(source_pixel_coords,
                                     [batch_size, 2, img_height, img_width])
    source_pixel_coords = tf.transpose(source_pixel_coords, perm=[0, 2, 3, 1])
    projected_img, mask = _spatial_transformer(img, source_pixel_coords)
    return projected_img, mask

def get_transform_mat(egomotion_vecs, i, j):
    """Returns a transform matrix defining the transform from frame i to j."""
    egomotion_transforms = []
    batchsize = tf.shape(egomotion_vecs)[0]
    if i == j:
        return tf.tile(tf.expand_dims(tf.eye(4, 4), axis=0), [batchsize, 1, 1])
    for k in range(min(i, j), max(i, j)):
        transform_matrix = _egomotion_vec2mat(egomotion_vecs[:, k, :], batchsize)
        if i > j:  # Going back in sequence, need to invert egomotion.
            egomotion_transforms.insert(0, tf.linalg.inv(transform_matrix))
        else:  # Going forward in sequence
            egomotion_transforms.append(transform_matrix)

    # Multiply all matrices.
    egomotion_mat = egomotion_transforms[0]
    for i in range(1, len(egomotion_transforms)):
        egomotion_mat = tf.matmul(egomotion_mat, egomotion_transforms[i])
    return egomotion_mat

def _pixel2cam(depth, pixel_coords, intrinsic_mat_inv):
    """Transform coordinates in the pixel frame to the camera frame."""
    cam_coords = tf.matmul(intrinsic_mat_inv, pixel_coords) * depth
    return cam_coords

def _cam2pixel(cam_coords, proj_c2p):
    """Transform coordinates in the camera frame to the pixel frame."""
    pcoords = tf.matmul(proj_c2p, cam_coords)
    x = tf.slice(pcoords, [0, 0, 0], [-1, 1, -1])
    y = tf.slice(pcoords, [0, 1, 0], [-1, 1, -1])
    z = tf.slice(pcoords, [0, 2, 0], [-1, 1, -1])
    # Not tested if adding a small number is necessary
    x_norm = x / (z + 1e-10)
    y_norm = y / (z + 1e-10)
    pixel_coords = tf.concat([x_norm, y_norm], axis=1)
    return pixel_coords

def _meshgrid_abs(height, width):
    """Meshgrid in the absolute coordinates."""
    x_t = tf.matmul(
            tf.ones(shape=tf.stack([height, 1])),
            tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
    y_t = tf.matmul(
            tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
            tf.ones(shape=tf.stack([1, width])))
    x_t = (x_t + 1.0) * 0.5 * tf.cast(width - 1, tf.float32)
    y_t = (y_t + 1.0) * 0.5 * tf.cast(height - 1, tf.float32)
    x_t_flat = tf.reshape(x_t, (1, -1))
    y_t_flat = tf.reshape(y_t, (1, -1))
    ones = tf.ones_like(x_t_flat)
    grid = tf.concat([x_t_flat, y_t_flat, ones], axis=0)
    return grid

def _create_intrinsics_matrix(input_image_shape, fov, x_scaling=1.0, y_scaling=1.0):
    """Create intrinsics matrix from given camera fov.

    Inputs:
        input_image_shape: Original input image shape.
        fov: Camera Field of View (in deg).
    Returns:
        intrinsics_mat: Intrinsics matrix [3x3].
    """
    F = input_image_shape[1] / (2 * np.tan((fov / 2) * np.pi / 180))
    print('[*] Creating intrinsics matrix from given FOV:', fov)

    # Create intrinsics matrix
    intrinsics_mat = np.array(
        [[F * x_scaling, 0, (input_image_shape[1] / 2) * x_scaling], 
            [0, F * y_scaling, (input_image_shape[0] / 2) * y_scaling],
            [0, 0, 1]])
    return intrinsics_mat


def _euler2mat(z, y, x):
    """Converts euler angles to rotation matrix.

     From:
     https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174

     TODO: Remove the dimension for 'N' (deprecated for converting all source
     poses altogether).

    Args:
        z: rotation angle along z axis (in radians) -- size = [B, n]
        y: rotation angle along y axis (in radians) -- size = [B, n]
        x: rotation angle along x axis (in radians) -- size = [B, n]

    Returns:
        Rotation matrix corresponding to the euler angles, with shape [B, n, 3, 3].
    """
    batch_size = tf.shape(z)[0]
    n = 1
    z = tf.clip_by_value(z, -np.pi, np.pi)
    y = tf.clip_by_value(y, -np.pi, np.pi)
    x = tf.clip_by_value(x, -np.pi, np.pi)

    # Expand to B x N x 1 x 1
    z = tf.expand_dims(tf.expand_dims(z, -1), -1)
    y = tf.expand_dims(tf.expand_dims(y, -1), -1)
    x = tf.expand_dims(tf.expand_dims(x, -1), -1)

    zeros = tf.zeros([batch_size, n, 1, 1])
    ones = tf.ones([batch_size, n, 1, 1])

    cosz = tf.cos(z)
    sinz = tf.sin(z)
    rotz_1 = tf.concat([cosz, -sinz, zeros], axis=3)
    rotz_2 = tf.concat([sinz, cosz, zeros], axis=3)
    rotz_3 = tf.concat([zeros, zeros, ones], axis=3)
    zmat = tf.concat([rotz_1, rotz_2, rotz_3], axis=2)

    cosy = tf.cos(y)
    siny = tf.sin(y)
    roty_1 = tf.concat([cosy, zeros, siny], axis=3)
    roty_2 = tf.concat([zeros, ones, zeros], axis=3)
    roty_3 = tf.concat([-siny, zeros, cosy], axis=3)
    ymat = tf.concat([roty_1, roty_2, roty_3], axis=2)

    cosx = tf.cos(x)
    sinx = tf.sin(x)
    rotx_1 = tf.concat([ones, zeros, zeros], axis=3)
    rotx_2 = tf.concat([zeros, cosx, -sinx], axis=3)
    rotx_3 = tf.concat([zeros, sinx, cosx], axis=3)
    xmat = tf.concat([rotx_1, rotx_2, rotx_3], axis=2)

    return tf.matmul(tf.matmul(xmat, ymat), zmat)

def _egomotion_vec2mat(vec, batch_size):
    """Converts 6DoF transform vector to transformation matrix.

    Args:
        vec: 6DoF parameters [tx, ty, tz, rx, ry, rz] -- [B, 6].
        batch_size: Batch size.

    Returns:
        A transformation matrix -- [B, 4, 4].
    """
    translation = tf.slice(vec, [0, 0], [-1, 3])
    translation = tf.expand_dims(translation, -1)
    rx = tf.slice(vec, [0, 3], [-1, 1])
    ry = tf.slice(vec, [0, 4], [-1, 1])
    rz = tf.slice(vec, [0, 5], [-1, 1])
    rot_mat = _euler2mat(rz, ry, rx)
    rot_mat = tf.squeeze(rot_mat, axis=[1])
    filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
    filler = tf.tile(filler, [batch_size, 1, 1])
    transform_mat = tf.concat([rot_mat, translation], axis=2)
    transform_mat = tf.concat([transform_mat, filler], axis=1)
    return transform_mat

def _bilinear_sampler(im, x, y, name='blinear_sampler'):
    """Perform bilinear sampling on im given list of x, y coordinates.

    Implements the differentiable sampling mechanism with bilinear kernel
    in https://arxiv.org/abs/1506.02025.

    x,y are tensors specifying normalized coordinates [-1, 1] to be sampled on im.
    For example, (-1, -1) in (x, y) corresponds to pixel location (0, 0) in im,
    and (1, 1) in (x, y) corresponds to the bottom right pixel in im.

    Args:
        im: Batch of images with shape [B, h, w, channels].
        x: Tensor of normalized x coordinates in [-1, 1], with shape [B, h, w, 1].
        y: Tensor of normalized y coordinates in [-1, 1], with shape [B, h, w, 1].
        name: Name scope for ops.

    Returns:
        Sampled image with shape [B, h, w, channels].
        Principled mask with shape [B, h, w, 1], dtype:float32.  A value of 1.0
            in the mask indicates that the corresponding coordinate in the sampled
            image is valid.
    """
    with tf.compat.v1.variable_scope(name):
        x = tf.reshape(x, [-1])
        y = tf.reshape(y, [-1])

        # Constants.
        batch_size = tf.shape(im)[0]
        _, height, width, channels = im.get_shape().as_list()

        x = tf.cast(x, dtype=tf.float32)
        y = tf.cast(y, dtype=tf.float32)
        height_f = tf.cast(height, 'float32')
        width_f = tf.cast(width, 'float32')
        zero = tf.constant(0, dtype=tf.int32)
        max_y = tf.cast(tf.shape(im)[1] - 1, 'int32')
        max_x = tf.cast(tf.shape(im)[2] - 1, 'int32')

        # Scale indices from [-1, 1] to [0, width - 1] or [0, height - 1].
        x = (x + 1.0) * (width_f - 1.0) / 2.0
        y = (y + 1.0) * (height_f - 1.0) / 2.0

        # Compute the coordinates of the 4 pixels to sample from.
        x0 = tf.cast(tf.floor(x), 'int32')
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), 'int32')
        y1 = y0 + 1

        mask = tf.logical_and(
                tf.logical_and(x0 >= zero, x1 <= max_x),
                tf.logical_and(y0 >= zero, y1 <= max_y))
        mask = tf.cast(mask, dtype=tf.float32)

        x0 = tf.clip_by_value(x0, zero, max_x)
        x1 = tf.clip_by_value(x1, zero, max_x)
        y0 = tf.clip_by_value(y0, zero, max_y)
        y1 = tf.clip_by_value(y1, zero, max_y)
        dim2 = width
        dim1 = width * height

        # Create base index.
        base = tf.range(batch_size) * dim1
        base = tf.reshape(base, [-1, 1])
        base = tf.tile(base, [1, height * width])
        base = tf.reshape(base, [-1])

        base_y0 = base + y0 * dim2
        base_y1 = base + y1 * dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # Use indices to lookup pixels in the flat image and restore channels dim.
        im_flat = tf.reshape(im, tf.stack([-1, channels]))
        im_flat = tf.cast(im_flat, dtype=tf.float32)
        pixel_a = tf.gather(im_flat, idx_a)
        pixel_b = tf.gather(im_flat, idx_b)
        pixel_c = tf.gather(im_flat, idx_c)
        pixel_d = tf.gather(im_flat, idx_d)

        x1_f = tf.cast(x1, dtype=tf.float32)
        y1_f = tf.cast(y1, dtype=tf.float32)

        # And finally calculate interpolated values.
        wa = tf.expand_dims(((x1_f - x) * (y1_f - y)), 1)
        wb = tf.expand_dims((x1_f - x) * (1.0 - (y1_f - y)), 1)
        wc = tf.expand_dims(((1.0 - (x1_f - x)) * (y1_f - y)), 1)
        wd = tf.expand_dims(((1.0 - (x1_f - x)) * (1.0 - (y1_f - y))), 1)

        output = tf.add_n([wa * pixel_a, wb * pixel_b, wc * pixel_c, wd * pixel_d])
        output = tf.reshape(output, tf.stack([batch_size, height, width, channels]))
        mask = tf.reshape(mask, tf.stack([batch_size, height, width, 1]))
        return output, mask

def _spatial_transformer(img, coords):
    """A wrapper over binlinear_sampler(), taking absolute coords as input."""
    img_height = tf.cast(tf.shape(img)[1], tf.float32)
    img_width = tf.cast(tf.shape(img)[2], tf.float32)
    px = coords[:, :, :, :1]
    py = coords[:, :, :, 1:]
    # Normalize coordinates to [-1, 1] to send to _bilinear_sampler.
    px = px / (img_width - 1) * 2.0 - 1.0
    py = py / (img_height - 1) * 2.0 - 1.0
    output_img, mask = _bilinear_sampler(img, px, py)
    return output_img, mask

def get_cloud(depth, intrinsics_inv, name=None):
    """Convert depth map to 3D point cloud."""
    with tf.compat.v1.name_scope(name):
        dims = depth.shape.as_list()
        batch_size, img_height, img_width = dims[0], dims[1], dims[2]
        depth = tf.reshape(depth, [batch_size, 1, img_height * img_width])
        grid = _meshgrid_abs(img_height, img_width)
        grid = tf.tile(tf.expand_dims(grid, 0), [batch_size, 1, 1])
        cam_coords = _pixel2cam(depth, grid, intrinsics_inv)
        cam_coords = tf.transpose(cam_coords, [0, 2, 1])
        cam_coords = tf.reshape(cam_coords, [batch_size, img_height, img_width, 3])
        logging.info('depth -> cloud: %s', cam_coords)
        return cam_coords

