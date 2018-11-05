import numpy as np
from numpy.random import randint

import matplotlib.pyplot as plt
import csv

#from  multiprocessing.dummy import Pool as ThreadPool # THIS IS NOT RIGHT, single process multi-thread, not friendly to Python
from multiprocessing import Pool as ProcessPool # This is real multi-process/multi-thread.

import time

from augmentation.dilate import *
from misc.lut import colorlut 
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

threads = 6

print("Visualization:", str(VISUALIZATION_ON))

cells = []
time_start = time.time()

myaug = MyAugmentor()

if VISUALIZATION_ON:
    for idx in random_idx:
        # Sequential execution
        image, masks_aug_d = myaug.exec_augment(all_rows[idx])

        image_aug, segmaps_aug_on_image = myaug.visualize()
        # Augmented segmap on augmented image
        for i in range(segmaps_aug_on_image.shape[0]):
            cells.append(segmaps_aug_on_image[i])

        cells.append(255 * masks_aug_d.transpose((1, 2, 0)))
else: 
    rows = [all_rows[idx] for idx in random_idx]
    # Parallel execution
    pool = ProcessPool(threads)
    results = pool.map(myaug.exec_augment, rows)
    pool.close()
    pool.join()

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
