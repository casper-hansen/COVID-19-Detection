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
            if avg < 40:
                num_of_black_pixels += 1
            #print(f'{k+1} {pixel} {avg} {num_of_black_pixels}')
        
        if num_of_black_pixels > len(row)/2:
            rows_to_remove.append(j)

    for transposed_img in img.T:
        for j, col in enumerate(transposed_img):
            num_of_black_pixels = 0
            for k, pixel in enumerate(col):
                avg = np.average(pixel)
                if avg < 40:
                    num_of_black_pixels += 1
                #print(f'{k+1} {pixel} {avg} {num_of_black_pixels}')
            
            if num_of_black_pixels > len(col)/2:
                cols_to_remove.append(j)

    image_rows_to_remove[i] = rows_to_remove
    image_cols_to_remove[i] = cols_to_remove

new_images = []

print(image_cols_to_remove)

for i, img in enumerate(images):
    img = np.delete(img, image_rows_to_remove[i], axis=0)
    img = np.delete(img, image_cols_to_remove[i], axis=1)
    print(img.shape)
    new_images.append(img)

plt.imshow(new_images[0])
plt.show()