import numpy as np
from numpy.random import randint

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import cv2
import csv
import ast

import time

from dilate import *
from lut import colorlut 

from PIL import Image, ImageDraw
import imgaug as ia
import imgaug.augmenters as iaa
import imageio
 

EXPORT_PATH = './data/combined-jpg'
all_rows = []
with open('tmp.csv', 'r', newline='') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',',
                       quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for row in csvreader:
        all_rows.append(row)


# Randomly pick several images
NUM_IMAGES_PER_ROUND = 5
DILATION_ON = True
DILATE_PIXELS = 2
random_idx = randint(0, len(all_rows), NUM_IMAGES_PER_ROUND)

images = []
segmaps_car = []
segmaps_rider = []
segmaps_ped = []
masks_car = []
masks_rider = []
masks_ped = []

raw_width = 1920
raw_height = 1208

for idx in random_idx:
    frame_file_name, tags, box_ids, box_classes, box_vertices,\
        poly_ids, poly_classes, poly_vertices = all_rows[idx]

    len_box = 0
    len_poly = 0
    
    # if box_ids is not None:
    if box_ids != '':
        box_ids = (box_ids).split(',')
        box_classes = (box_classes).split(',')
        #box_attrs = (box_attrs).split(',')
        box_vertices = ast.literal_eval(box_vertices)
        if len(box_ids) == 1: # Special case handling
            box_vertices = [box_vertices]
        assert len(box_ids) == len(box_classes) 
        if len(box_ids)!=1:
            assert len(box_ids) == len(box_vertices)
        len_box = len(box_ids)
            
    # if poly_ids is not None:
    if poly_ids != '':
        poly_ids = (poly_ids).split(',')
        poly_classes = (poly_classes).split(',')
        #poly_attrs = (poly_attrs).split(',')
        poly_vertices = ast.literal_eval(poly_vertices)
        if len(poly_ids) == 1: # Special case handling
            poly_vertices = [poly_vertices]
        assert len(poly_ids) == len(poly_classes)
        if len(poly_ids)!=1:
            assert len(poly_ids) == len(poly_vertices)
        len_poly = len(poly_ids)
    
    
    # Frame associated with the labels grouped
    # Override using .jpg files
    format_ext = ".jpg"
    frame_file_name = "".join([str(frame_file_name), format_ext])
    frame_file = "".join([EXPORT_PATH, "/", frame_file_name])
    print("idx, frame_file:", idx, frame_file)
    myimg = cv2.imread(frame_file)
    myimg_chw = myimg.transpose(2, 0, 1)/255.
    
    # Create polygon masks
   
    # Override with (272, 480)
    format_width_scale = 480./raw_width
    format_height_scale = 272./raw_height
    width = int(raw_width * format_width_scale)
    height = int(raw_height * format_height_scale) 
    
    
    mask_car = np.zeros((height, width))
    mask_rider = np.zeros((height, width))
    mask_ped = np.zeros((height, width))
    
    # Creating segmentation mask
   
    # Iterate through bbox classes and create mask using imgaug
    # if box_ids is not None:
    if box_ids != '':
        for i in range(len_box): 
            # class mapped color/index
            box_color_idx_int = colorlut[box_classes[i]]
            if box_color_idx_int == 0 or box_color_idx_int > 3:
                print("Wired class in BBOX:", box_classes[i])
                continue

            # Empty message to start with
            tmp_img = Image.new('L', (width, height), 0)
    
            # Expand box((x0, y0), (x1, y1) to ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
            x0, y0, x1, y1 = (np.array(box_vertices[i])).flatten()
    
            x0 *= format_width_scale
            x1 *= format_width_scale
            y0 *= format_height_scale
            y1 *= format_height_scale
            
            # Draw current polygons
            j = len_poly + i
            ImageDraw.Draw(tmp_img).polygon(
                [x0, y0, x1, y0, x1, y1, x0, y1], 
                outline=1, fill=1
            )
    
            tmp_mask = np.array(tmp_img)
            # Overlay onto mask
            if box_color_idx_int == 1: # Car
                mask_car = np.maximum(tmp_mask, mask_car)
            elif box_color_idx_int == 2: # Rider
                mask_rider = np.maximum(tmp_mask, mask_rider)
            elif box_color_idx_int == 3: # Ped
                mask_ped = np.maximum(tmp_mask, mask_ped)
    
    # !!! NOTICE: latter mask will overwrite previous ones!!! 
    # Iterate through polygon classes and create mask using imgaug
    # if poly_ids is not None:
    if poly_ids != '':
        for i in range(len_poly):
            # class mapped color/index
            poly_color_idx_int = colorlut[poly_classes[i]]
            if poly_color_idx_int == 0 or poly_color_idx_int > 3:
                print("Wired class in POLYGON:", poly_classes[i])
                continue

            # Empty message to start with
            tmp_img = Image.new('L', (width, height), 0)
    
            scaled_list = [
                p*format_width_scale if i%2 else p*format_height_scale 
                for i,p in enumerate(list(
                    (np.array(poly_vertices[i])).flatten()), 1)
            ]
            # Draw current bbox 
            ImageDraw.Draw(tmp_img).polygon(
                list((np.array(poly_vertices[i])).flatten()),
                outline=1, fill=1
            )
    
            tmp_mask = np.array(tmp_img)
            # Overlay onto mask
            if box_color_idx_int == 1: # Car
                mask_car = np.maximum(tmp_mask, mask_car)
            elif box_color_idx_int == 2: # Rider
                mask_rider = np.maximum(tmp_mask, mask_rider)
            elif box_color_idx_int == 3: # Ped
                mask_ped = np.maximum(tmp_mask, mask_ped)
     
    segmap_car = ia.SegmentationMapOnImage(np.uint8(mask_car), shape=(height, width),
                    nb_classes=1+np.int(np.max(mask_car)))
    segmap_rider = ia.SegmentationMapOnImage(np.uint8(mask_rider), shape=(height, width),
                    nb_classes=1+np.int(np.max(mask_rider)))
    segmap_ped = ia.SegmentationMapOnImage(np.uint8(mask_ped), shape=(height, width),
                    nb_classes=1+np.int(np.max(mask_ped)))
    
    # Visualize the mask image
    #segmap_np_array = segmap.draw()
    #plt.imshow(segmap_np_array)
    #plt.show()
    
    # Visualize mask overlayed onto origin image
    #segmap_chw = segmap_np_array.transpose(2, 0, 1)
    #dispimg = cv2.addWeighted(np.uint8(myimg_chw*255), 0.5, segmap_chw, 0.5, 0)
    #plt.imshow(dispimg.transpose(1, 2, 0))
    #plt.show()
    
    # Load an image (uint8) for later augmentation.
    images.append(cv2.cvtColor(myimg, cv2.COLOR_BGR2RGB))
    segmaps_car.append(segmap_car)
    segmaps_rider.append(segmap_rider)
    segmaps_ped.append(segmap_ped)
    masks_car.append(mask_car)
    masks_rider.append(mask_rider)
    masks_ped.append(mask_ped)

# Reverse the list
images.reverse()
segmaps_car.reverse()
segmaps_rider.reverse()
segmaps_ped.reverse()
masks_car.reverse()
masks_rider.reverse()
masks_ped.reverse()

# Augmentation
ia.seed(1)

# Define our augmentation pipeline.
seq = iaa.Sequential([
    # iaa.Fliplr(0.5), # 50% change horizontal flipping
    iaa.Add((-20, 20)), # Add random values between -20 and 20 to images
    iaa.ContrastNormalization((0.9, 1.1)), # Normalize contrast by a factor of 0.9 to 1.1
    # RGB->HSV, random shift, then HSV->RGB
    iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
    iaa.WithChannels(0, iaa.Add((0, 30))),
    iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB"),
    # Affine
    iaa.Affine(
        rotate=(-10, 10),  # rotate by -10 to 10 degrees (affects heatmaps)
        translate_px={"x": (-50, 50), "y": (-25, 25)}, # translation in pixels
        scale={"x": (0.9, 1.1), "y": (0.9, 1.1)}, # Scaling +/-20%
        shear=(-5, 5) # Shear by +/-5%
    )
], random_order=False)

# Augment images and heatmaps.
images_aug = []
segmaps_car_aug = []
segmaps_rider_aug = []
segmaps_ped_aug = []
for i in range(NUM_IMAGES_PER_ROUND):
    seq_det = seq.to_deterministic()
    images_aug.append(seq_det.augment_image(images[i]))
    segmaps_car_aug.append(seq_det.augment_segmentation_maps([segmaps_car[i]])[0])
    segmaps_rider_aug.append(seq_det.augment_segmentation_maps([segmaps_rider[i]])[0])
    segmaps_ped_aug.append(seq_det.augment_segmentation_maps([segmaps_ped[i]])[0])

mymasks = []
mysegmap_augs = []
mysegmap_aug_on_images = []
for image_aug, segmap_car_aug, segmap_rider_aug, segmap_ped_aug, mask_car, mask_rider, mask_ped \
        in zip(images_aug, segmaps_car_aug, segmaps_rider_aug, segmaps_ped_aug, masks_car, masks_rider, masks_ped):
    # Collect masks
    mymasks.append(mask_car)
    mymasks.append(mask_rider)
    mymasks.append(mask_ped)

    # Collect augmented segmaps
    tmp_segmap_aug = segmap_car_aug.draw(size=image_aug.shape[:2])
    mysegmap_augs.append(tmp_segmap_aug)
    tmp_segmap_aug = segmap_rider_aug.draw(size=image_aug.shape[:2])
    mysegmap_augs.append(tmp_segmap_aug)
    tmp_segmap_aug = segmap_ped_aug.draw(size=image_aug.shape[:2])
    mysegmap_augs.append(tmp_segmap_aug)

    # Collect augmented segmaps on augmented images
    tmp_segmap_aug_on_image = segmap_car_aug.draw_on_image(image_aug)
    mysegmap_aug_on_images.append(tmp_segmap_aug_on_image)
    tmp_segmap_aug_on_image = segmap_rider_aug.draw_on_image(image_aug)
    mysegmap_aug_on_images.append(tmp_segmap_aug_on_image)
    tmp_segmap_aug_on_image = segmap_ped_aug.draw_on_image(image_aug)
    mysegmap_aug_on_images.append(tmp_segmap_aug_on_image)


# Masks only
threads = 8
   
# Parallel version
cells = []
time_start = time.time()

if DILATION_ON:
    results  = dilate_k_pix_threads(mymasks, threads)
    results_2  = dilate_k_pix_threads(mysegmap_augs, threads)
else:
    results = mymasks
    results_2 = mysegmap_augs

for i in range(NUM_IMAGES_PER_ROUND):
    idx = 3*i

    # mask_car = mymasks[idx]
    # mask_rider = mymasks[idx+1]
    # mask_ped = mymasks[idx+2]

    # mask_car_d = results[idx]
    # mask_rider_d = results[idx+1]
    # mask_ped_d = results[idx+2]

    segmap_car_aug_d = results_2[idx]
    segmap_rider_aug_d = results_2[idx+1]
    segmap_ped_aug_d = results_2[idx+2]

    # cells.append(images[i])
    # Concatenate and add a new dimension at tail
    # cells.append(255 * np.stack([mask_car, mask_rider, mask_ped], axis=2)) 

    # Individual dilated mask
    # cells.append(255 * np.stack([mask_car_d, mask_rider_d, mask_ped_d], axis=2))
    # # Augmented image
    # cells.append(images_aug[i])
    # # Individual dilated augmented segmap
    # cells.append(255 * segmap_car_aug_d)
    # cells.append(255 * segmap_rider_aug_d)
    # cells.append(255 * segmap_ped_aug_d)
    # Augmented segmap on augmented image
    cells.append(mysegmap_aug_on_images[idx])
    cells.append(mysegmap_aug_on_images[idx+1])
    cells.append(mysegmap_aug_on_images[idx+2])

    # Convert augmented segmap to augmented masks
    #   segmap_xxx_aug_d is in shape (272, 480, 3), an array of True/False
    mask_car_aug_d = np.any(segmap_car_aug_d, axis=2).astype(np.float)
    mask_rider_aug_d = np.any(segmap_rider_aug_d, axis=2).astype(np.float)
    mask_ped_aug_d = np.any(segmap_ped_aug_d, axis=2).astype(np.float)
    cells.append(255 * np.stack([mask_car_aug_d, mask_rider_aug_d, mask_ped_aug_d], axis=2))


time_end = time.time()
print("time spent (parallel threads version):", time_end - time_start)

# Convert cells to grid image and save.
grid_image = ia.draw_grid(cells, cols=int(len(cells)/NUM_IMAGES_PER_ROUND))
plt.imshow(grid_image)
# figM = plt.get_current_fig_manager()
# figM.resize(*figM.window.maxsize())
plt.show()

imageio.imwrite("example_segmaps.jpg", grid_image)
