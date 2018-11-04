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



def parse_row(row):
    """
    Parse a row read from label record CSV file. Deal with some anomaly cases in 
    the meantime. 
    
    Return these parameters:
        frame_file_name, tags, len_box, box_ids, box_classes, box_vertices,
        len_poly, poly_ids, poly_classes, poly_vertices
    """
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

    return frame_file_name, tags, len_box, box_ids, box_classes, box_vertices,\
        len_poly, poly_ids, poly_classes, poly_vertices
  

def mask_bbox(mask_car, mask_rider, mask_ped, \
              len_box, box_ids, box_classes, box_vertices):
    """
    Create binary [0, 1] mask image from bounding box labels.
        mask_xxx = np.zeros((height, width))
    """
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
            ImageDraw.Draw(tmp_img).polygon(
                [x0, y0, x1, y0, x1, y1, x0, y1], 
                outline=1, fill=1 # Choose to use value 1 for all classes
            )
    
            tmp_mask = np.array(tmp_img)
            # Overlay onto mask
            if box_color_idx_int == 1: # Car
                mask_car = np.maximum(tmp_mask, mask_car)
            elif box_color_idx_int == 2: # Rider
                mask_rider = np.maximum(tmp_mask, mask_rider)
            elif box_color_idx_int == 3: # Ped
                mask_ped = np.maximum(tmp_mask, mask_ped)
    
    return mask_car, mask_rider, mask_ped


def mask_polygon(mask_car, mask_rider, mask_ped, \
              len_poly, poly_ids, poly_classes, poly_vertices):
    """
    Create binary [0, 1] mask image from polygon labels.
        mask_xxx = np.zeros((height, width))
    """
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
            if poly_color_idx_int == 1: # Car
                mask_car = np.maximum(tmp_mask, mask_car)
            elif poly_color_idx_int == 2: # Rider
                mask_rider = np.maximum(tmp_mask, mask_rider)
            elif poly_color_idx_int == 3: # Ped
                mask_ped = np.maximum(tmp_mask, mask_ped)

    return mask_car, mask_rider, mask_ped


def gen_segmaps(cur_row, height, width):
    """
    Provided the CSV label record row, generate segmaps for labels. 
    """
    frame_file_name, tags, len_box, box_ids, box_classes, box_vertices,\
        len_poly, poly_ids, poly_classes, poly_vertices = parse_row(cur_row)
   
    # Create polygon masks
    mask_car = np.zeros((height, width))
    mask_rider = np.zeros((height, width))
    mask_ped = np.zeros((height, width))
    
    # Creating segmentation mask
   
    # Iterate through bbox classes and create mask using imgaug
    mask_car, mask_rider, mask_ped = mask_bbox(mask_car, mask_rider, mask_ped, \
              len_box, box_ids, box_classes, box_vertices)

    # !!! NOTICE: latter mask will overwrite previous ones!!! 
    # Iterate through polygon classes and create mask using imgaug
    mask_car, mask_rider, mask_ped = mask_polygon(mask_car, mask_rider, mask_ped, \
              len_poly, poly_ids, poly_classes, poly_vertices)

    # Convert to object representing a segmentation map associated with an image.
    #   Necessary for later augmentation.
    segmap_car = ia.SegmentationMapOnImage(np.uint8(mask_car), shape=(height, width),
                    nb_classes=1+np.int(np.max(mask_car)))
    segmap_rider = ia.SegmentationMapOnImage(np.uint8(mask_rider), shape=(height, width),
                    nb_classes=1+np.int(np.max(mask_rider)))
    segmap_ped = ia.SegmentationMapOnImage(np.uint8(mask_ped), shape=(height, width),
                    nb_classes=1+np.int(np.max(mask_ped)))
    
    return frame_file_name, segmap_car, segmap_rider, segmap_ped


EXPORT_PATH = './data/combined-jpg'
all_rows = []
with open('./data/dataset.csv', 'r', newline='') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',',
                       quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for row in csvreader:
        all_rows.append(row)


width = 480
height = 272
raw_width = 1920
raw_height = 1208
format_width_scale = float(width)/raw_width
format_height_scale = float(height)/raw_height

# Randomly pick several images
NUM_IMAGES_PER_ROUND = 5
DILATION_ON = True
DILATE_PIXELS = 2
random_idx = randint(0, len(all_rows), NUM_IMAGES_PER_ROUND)

images = []
segmaps_car = []
segmaps_rider = []
segmaps_ped = []

mysegmap_augs = []
mysegmap_aug_on_images = []

for idx in random_idx:

    frame_file_name, segmap_car, segmap_rider, segmap_ped = \
        gen_segmaps(all_rows[idx], height, width)

    # Frame associated with the labels grouped
    # Override using .jpg files
    format_ext = ".jpg"
    frame_file_name = "".join([str(frame_file_name), format_ext])
    frame_file = "".join([EXPORT_PATH, "/", frame_file_name])
    print("idx, frame_file:", idx, frame_file)
    myimg = cv2.imread(frame_file)
    myimg_chw = myimg.transpose(2, 0, 1)/255.

     # Load an image (uint8) for later augmentation.
    image = cv2.cvtColor(myimg, cv2.COLOR_BGR2RGB)

    seq_det = seq.to_deterministic()

    # Augment images and heatmaps.
    image_aug = seq_det.augment_image(image)
    segmap_car_aug = seq_det.augment_segmentation_maps([segmap_car])[0]
    segmap_rider_aug = seq_det.augment_segmentation_maps([segmap_rider])[0]
    segmap_ped_aug = seq_det.augment_segmentation_maps([segmap_ped])[0]

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
    results_2  = dilate_k_pix_threads(mysegmap_augs, threads)
else:
    results_2 = mysegmap_augs

for i in range(NUM_IMAGES_PER_ROUND):
    idx = 3*i

    segmap_car_aug_d = results_2[idx]
    segmap_rider_aug_d = results_2[idx+1]
    segmap_ped_aug_d = results_2[idx+2]

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
