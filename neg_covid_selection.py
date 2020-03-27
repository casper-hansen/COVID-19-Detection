import pandas as pd
import matplotlib.pyplot as plt
import cv2, os
import numpy as np
import random

df = pd.read_csv('kaggle-pneumonia-jpg/stage_2_detailed_class_info.csv')
image_folder = 'kaggle-pneumonia-jpg/stage_2_train_images_jpg/'

df = df.loc[df['class'] == 'Normal']

images = []
labels = []
filenames = []

for index, row in df.iterrows():
    # cv2 loads BGR colors by default, switch them to RGB
    # resize the image to 224, 224
    image = cv2.imread(image_folder + row.patientId + '.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    images.append(image)

    label = row['class']
    labels.append(label)
    
    filename = row.patientId
    filenames.append(filename)

save_n_images = 1000
save_folder = 'negative_dataset/'
collection_of_ran_num = []

new_filename = []
new_finding = []

while save_n_images > 0:
    ran_num = random.randint(0, len(images))
    if ran_num in collection_of_ran_num:
        continue
    else:
        collection_of_ran_num.append(ran_num)
        save_n_images -= 1

    image = images[ran_num]
    label = labels[ran_num]
    filename = filenames[ran_num]

    cv2.imwrite(save_folder + filename + '.jpg', image)
    new_filename.append(filename + '.jpg')
    new_finding.append(label)

new_df = pd.DataFrame({'filename': new_filename, 'finding': new_finding})
new_df.to_csv('normal_xray_dataset.csv')