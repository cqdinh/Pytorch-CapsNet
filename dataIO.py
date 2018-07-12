import cv2
import numpy as np
import os
import random
import torch
from torch.utils.data import Dataset
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from skimage.color import rgb2gray, gray2rgb


# Use the graham scan algorithm to find a convex hull of points
def graham_scan(points):
    def cross(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])

    # Computes slope of line between p1 and p2
    def slope(p1, p2):
        return 1.0 * (p1[1] - p2[1]) / (p1[0] - p2[0]) if p1[0] != p2[0] else float('inf')

    # Find the smallest left point and remove it from points
    start = min(points, key=lambda p: (p[0], p[1]))
    points.pop(points.index(start))

    # Sort points so that traversal is from start in a ccw circle.
    points.sort(key=lambda p: (slope(p, start), -p[1], p[0]))

    # Add each point to the convex hull.
    # If the last 3 points make a cw turn, the second to last point is wrong.
    ans = [start]
    for p in points:
        ans.append(p)
        while len(ans) > 2 and cross(ans[-3], ans[-2], ans[-1]) < 0:
            ans.pop(-2)

    return ans


# Determine if the corners are in an acceptable range
# This prevents the logo from being overly distorted
def goodangles(corners, min_angle, max_angle):
    def angle(x1, y1, x2, y2, x3, y3):
        ab = (x1 - x2, y1 - y2)
        bc = (x3 - x2, y3 - y2)
        dot = ab[0] * bc[0] + ab[1] * bc[1]
        norm_ab_square = ab[0] * ab[0] + ab[1] * ab[1]
        norm_bc_square = bc[0] * bc[0] + bc[1] * bc[1]
        norm = np.sqrt(norm_ab_square * norm_bc_square)
        return np.arccos(dot / norm)

    a = corners[2]
    b = corners[3]
    c = corners[0]
    i = 3
    while i >= 0:
        corner_angle = angle(a[0], a[1], b[0], b[1], c[0], c[1])
        if not min_angle < corner_angle < max_angle:
            return False
        c = b
        b = a
        a = corners[i - 2]
        i -= 1
    return True


# A pytorch Dataset that generates randomly transformed versions of logos
class DataGen(Dataset):
    def __init__(self,
                 logos_folder,
                 backgrounds_folder,
                 data_size,
                 transform_bounds,
                 epoch_size,
                 grayscale=False,
                 use_background=True):
        '''
        :param logos_folder: The path to the folder containing the logo files
        :param backgrounds_folder: The path to the folder containing the background images to place the logos on
        :param data_size: The width and height of the output images
        :param transform_bounds: The size of the box that the transformed logo must fit in
        :param epoch_size: The number of samples to include in each epoch
        :param grayscale: Whether or not to convert the images to grayscale
        :param use_background: Whether or not to place the transformed logo onto a patch of a background image
        '''
        self.use_background = use_background
        self.epoch_size = epoch_size
        self.logos = []
        self.labels = []
        self.data_size = data_size
        self.transform_bound = transform_bounds
        self.grayscale = grayscale

        # Load the logo images
        # The logo name is just the filename
        # The [..., ::-1] is there to change the images from BGR to RGB because OpenCV loads them in as BGR
        for file in os.listdir(logos_folder):
            logo = cv2.imread(os.path.join(logos_folder, file))[..., ::-1].astype(np.float64)

            logo = cv2.resize(logo, (transform_bounds, transform_bounds))
            if grayscale:
                logo = rgb2gray(logo)
                logo = np.expand_dims(logo, -1)

            self.logos.append(logo)
            self.labels.append(os.path.splitext(file)[0])

        # Set the one-hot representation of each label
        self.one_hot = [np.zeros(len(self.labels)) for i in range(len(self.labels))]
        for i in range(len(self.labels)):
            self.one_hot[i][i] = 1

        # Load the background images and reverse the channels to make them RGB because OpenCV loads BGR
        self.backgrounds = []
        for file in os.listdir(backgrounds_folder):
            background = cv2.imread(os.path.join(backgrounds_folder, file))[..., ::-1].astype(np.float64)
            if grayscale:
                background = np.expand_dims(rgb2gray(background), -1)
            self.backgrounds.append(background)

        self.num_categories = len(self.logos)

    # len and getitem exist to fit the Dataset template for pytorch
    def __len__(self):
        return self.epoch_size

    def __getitem__(self, item):
        return self.generate()

    # Generate an image with a distorted logo
    def generate(self):

        # Pick a random logo and get its one-hot label
        logo_index = random.randrange(0, len(self.logos))
        logo = self.logos[logo_index]
        one_hot_vec = self.one_hot[logo_index]

        # Pick a random background and sample a random patch from it
        background = random.choice(self.backgrounds)
        patch_x = np.random.randint(0, background.shape[1] - self.data_size)
        patch_y = np.random.randint(0, background.shape[0] - self.data_size)
        background = background[patch_y:patch_y + self.data_size, patch_x:patch_x + self.data_size]

        # Find the bounding box of the logo within the image
        col_sums = np.sum(np.sum(logo, 0), 1)
        left = 0
        right = 0
        for i in range(len(col_sums)):
            if col_sums[i]:
                right = i
                if not left:
                    left = i

        row_sums = np.sum(np.sum(logo, 1), 1)
        top = 0
        bottom = 0
        for i in range(len(row_sums)):
            if row_sums[i]:
                bottom = i
                if not top:
                    top = i

        # The initial corners of the logo
        src_corners = np.array([
            [left, top],
            [right, top],
            [right, bottom],
            [left, bottom]
        ], np.float32)

        # Generate the corners of the transformed logo

        # Repeatedly generate 4 random points until they form a convex hull and
        # all angles are between 60 and 120 degrees
        corners = []
        min_angle = 60 * np.pi / 180
        max_angle = 120 * np.pi / 180
        while len(corners) != 4 or not goodangles(corners, min_angle, max_angle):
            corners = graham_scan([
                                      (np.random.rand() * self.transform_bound, np.random.rand() * self.transform_bound)
                                      for i in range(4)
                                      ])

        # Randomly decide whether to flip the corners so that mirroring is possible
        if np.random.rand() > 0.5:
            corners = reversed(corners)

        # Convert the transformed logo corners into a numpy array
        corners = np.array([[corner[0], corner[1]] for corner in corners], np.float32)

        # Calculate the transformation matrix from the original corners to the transformed ones
        transform = cv2.getPerspectiveTransform(src_corners, corners)

        # Transform the logo image using this transformation matrix
        transformed_logo = cv2.warpPerspective(logo, transform, (self.transform_bound, self.transform_bound))

        # Crop the image to the data_size
        pad_size = (self.transform_bound - self.data_size) // 2
        extra_pixel = (self.transform_bound - self.data_size) % 2

        transformed_logo = transformed_logo[pad_size:self.transform_bound - pad_size - extra_pixel,
                                            pad_size:self.transform_bound - pad_size - extra_pixel]

        # Calculate a mask that is 255 wherever the transformed logo exists and 0 everywhere else
        logo_to_mask = transformed_logo if self.grayscale else rgb2gray(transformed_logo.astype(np.uint8))
        mask = cv2.threshold(logo_to_mask, 0.05, 255, cv2.THRESH_BINARY)[1]

        # Ensure that the logo takes up at least 10% of the image or 10 pixels whichever is smaller
        if np.sum(mask) / 255.0 < max(10, 0.1 * self.transform_bound * self.transform_bound):
            return self.generate()

        # Expand the mask to the correct number of dimensions
        if self.grayscale:
            mask = np.expand_dims(mask, -1)
        else:
            mask = gray2rgb(mask)

        # Remove the part of the background that is inside of the logo
        background = cv2.bitwise_and(background, cv2.bitwise_not(mask))

        # Use the mask to remove noisy nonzero pixels outside of the logo
        transformed_logo = cv2.bitwise_and(transformed_logo, mask)

        # Place the transformed logo on the background
        logo = background + transformed_logo

        # Convert the logo images with and without the background into pytorch tensors
        logo = torch.from_numpy(logo.astype(np.uint8)).type(torch.FloatTensor)
        transformed_logo = torch.from_numpy(transformed_logo.astype(np.uint8)).type(torch.FloatTensor)

        # Reshape the images appropriately
        if self.grayscale:
            logo = logo.unsqueeze(0)
            transformed_logo = transformed_logo.unsqueeze(0)
        else:
            logo = logo.permute(2, 0, 1)
            transformed_logo = transformed_logo.permute(2, 0, 1)

        # Normalize the images
        logo /= 255.0
        transformed_logo /= 255.0

        # Convert the logo index and one-hot label into pytorch tensors
        one_hot_vec = torch.from_numpy(one_hot_vec).type(torch.FloatTensor)
        logo_index = torch.from_numpy(np.array([logo_index])).type(torch.FloatTensor)

        # Ensure that there are nonzero values in the logo and there are no NaNs
        if len(torch.nonzero(logo).size()) == 0 or (logo != logo).any():
            return self.generate()

        # If the background is not going to be used, replace the image with the background with the image without
        if not self.use_background:
            logo = transformed_logo

        return logo, transformed_logo, one_hot_vec, logo_index
