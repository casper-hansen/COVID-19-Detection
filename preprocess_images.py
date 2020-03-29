import pandas as pd
import matplotlib.pyplot as plt
import cv2, os
import numpy as np
import random

df = pd.read_csv('kaggle-pneumonia-jpg/stage_2_detailed_class_info.csv')
image_folder = 'kaggle-pneumonia-jpg/stage_2_train_images_jpg/'

df = df.loc[df['class'] == 'Normal']
df = df.head(5)

images = []
labels = []
filenames = []

for index, row in df.iterrows():
    image = cv2.imread(image_folder + row.patientId + '.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    images.append(image)

    label = row['class']
    labels.append(label)
    
    filename = row.patientId
    filenames.append(filename)

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
    col_row_diff = len(image_cols_to_remove[i]) - len(image_rows_to_remove[i])
    row_col_diff = len(image_rows_to_remove[i]) - len(image_cols_to_remove[i])
    
    img = new_images[i]
    
    if col_row_diff > 0:
        slice_size = int(col_row_diff/2)
        
        img = img[:-slice_size]
        img = img[slice_size:]
        
        if img.shape[0] != img.shape[1]:
            img = img[:-1]
            
    elif row_col_diff > 0:
        slice_size = int(row_col_diff/2)
        
        img = img[:,:-slice_size,:]
        img = img[:,slice_size:,:]
        
        if img.shape[0] != img.shape[1]:
            img = img[:,:-1,:]
    
    if img.shape[0] == img.shape[1]:
        adjusted_images.append(img)
    