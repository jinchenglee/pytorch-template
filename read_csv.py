import numpy as np
from numpy.random import randint

import matplotlib.pyplot as plt
import csv

import time

from dilate import *
from lut import colorlut 
from augmentation.aug import MyAugmentor

import imgaug as ia
import imageio


EXPORT_PATH = './data/combined-jpg'
all_rows = []
time_start = time.time()
with open('./data/dataset.csv', 'r', newline='') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',',
                       quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for row in csvreader:
        all_rows.append(row)
time_end = time.time()
print("Reading CSV file... Time spent:", time_end - time_start)



VISUALIZATION_ON = True

# Randomly pick several images
NUM_IMAGES_PER_ROUND = 5
random_idx = randint(0, len(all_rows), NUM_IMAGES_PER_ROUND)


print("Visualization:", str(VISUALIZATION_ON))

cells = []
time_start = time.time()

for idx in random_idx:

    myaug = MyAugmentor()

    image, mask_car_aug_d, mask_rider_aug_d, mask_ped_aug_d = myaug.exec_augment(all_rows[idx])

    if VISUALIZATION_ON:

        image_aug, segmap_car_aug_on_image, \
            segmap_rider_aug_on_image, segmap_ped_aug_on_image = \
                myaug.visualize()
            
        # Augmented segmap on augmented image
        cells.append(segmap_car_aug_on_image)
        cells.append(segmap_rider_aug_on_image)
        cells.append(segmap_ped_aug_on_image)

        cells.append(255 * np.stack([mask_car_aug_d, mask_rider_aug_d, mask_ped_aug_d], axis=2))

time_end = time.time()
print("Visualization... Time spent:", time_end - time_start)



if VISUALIZATION_ON:

    # Convert cells to grid image and save.
    grid_image = ia.draw_grid(cells, cols=int(len(cells)/NUM_IMAGES_PER_ROUND))
    plt.imshow(grid_image)
    figM = plt.get_current_fig_manager()
    figM.resize(*figM.window.maxsize())
    plt.show()

    imageio.imwrite("example_segmaps.jpg", grid_image)
