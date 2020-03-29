import pandas as pd
import matplotlib.pyplot as plt
import cv2, os
import numpy as np
import random

folder = 'cropping_experiment/'
images_names = ['1.jpg', '2.jpg', '3.jpg', '4.jpg']

images = []

for img in images_names:
    image = cv2.imread(folder + img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    images.append(image)

images = np.array(images)

image_rows_to_remove = {}
image_cols_to_remove = {}

for i, img in enumerate(images):
    rows_to_remove = []
    cols_to_remove = []

    for j, row in enumerate(img):
        num_of_black_pixels = 0
        for k, pixel in enumerate(row):
            avg = np.average(pixel)
            if avg < 20:
                num_of_black_pixels += 1
        
        if num_of_black_pixels > len(row)/2:
            rows_to_remove.append(j)

    # good enough to only loop on one color channel (could be improved)
    for j, col in enumerate(img.T[0]):
        num_of_black_pixels = 0
        for k, pixel in enumerate(col):
            avg = np.average(pixel)
            if avg < 40:
                num_of_black_pixels += 1
        
        if num_of_black_pixels > len(col)/2:
            cols_to_remove.append(j)

    image_rows_to_remove[i] = rows_to_remove
    image_cols_to_remove[i] = cols_to_remove

for key, arr in image_rows_to_remove.items():
    last_val = None
    for i, value in enumerate(arr):
        if 200 <= value <= 800:
            del image_rows_to_remove[key][i]
        else:
            last_val = value
            
for key, arr in image_cols_to_remove.items():
    last_val = None
    for i, value in enumerate(arr):
        if 200 <= value <= 800:
            del image_cols_to_remove[key][i]
        else:
            last_val = value

new_images = []

for i, img in enumerate(images):
    img = np.delete(img, image_rows_to_remove[i], axis=0)
    img = np.delete(img, image_cols_to_remove[i], axis=1)
    new_images.append(img)

adjusted_images = []

for i in range(len(new_images)):
    n_rows_removed = len(image_rows_to_remove[i])
    n_cols_removed = len(image_cols_to_remove[i])
    row_col_diff = n_rows_removed - n_cols_removed
    col_row_diff = n_cols_removed - n_rows_removed
    
    if col_row_diff > 0:
        slice_size_bottom = int(col_row_diff/3)
        slice_size_top = int(col_row_diff/3)*2+1
        
        img = new_images[i][:-slice_size_top]
        img = img[slice_size_bottom:]
        
        if img.shape[0] != img.shape[1]:
            img = img[:-1]
        
        adjusted_images.append(img)
            
    elif row_col_diff > 0:
        pass
    
    if adjusted_images[i].shape[0] != adjusted_images[i].shape[1]:
        adjusted_images.pop()

