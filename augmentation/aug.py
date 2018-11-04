import numpy as np

import cv2
import ast

from dilate import *
from lut import colorlut 

from PIL import Image, ImageDraw
import imgaug as ia
import imgaug.augmenters as iaa


class MyAugmentor():
    """
    Data augmentation while data loading. Use imgaug library.
    """

    def __init__(self, width=480, height=272, dilation=True, dilate_k_pixels=2):

        self.dilation = dilation
        self.dilate_k_pixels = dilate_k_pixels

        self.format_ext = ".jpg"
        self.export_path = './data/combined-jpg'

        self.width = 480
        self.height = 272
        raw_width = 1920
        raw_height = 1208
        self.format_width_scale = float(self.width)/raw_width
        self.format_height_scale = float(self.height)/raw_height

        # Augmentation
        ia.seed(1)

        # Define our augmentation pipeline.
        self.seq = iaa.Sequential([
            # iaa.Fliplr(0.5), # 50% change horizontal flipping
            iaa.Add((-20, 20)), # Add random values between -20 and 20 to images
            iaa.ContrastNormalization((0.9, 1.1)), # Normalize contrast by a factor of 0.9 to 1.1
            # RGB->HSV, random shift, then HSV->RGB
            iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
            iaa.WithChannels(0, iaa.Add((0, 15))),
            iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB"),
            # Affine
            iaa.Affine(
                rotate=(-10, 10),  # rotate by -10 to 10 degrees (affects heatmaps)
                translate_px={"x": (-50, 50), "y": (-25, 25)}, # translation in pixels
                scale={"x": (0.9, 1.1), "y": (0.9, 1.1)}, # Scaling +/-20%
                shear=(-5, 5) # Shear by +/-5%
            )
        ], random_order=False)



    def _parse_row(self, cur_row):
        """
        Parse a row read from label record CSV file. Deal with some anomaly cases in 
        the meantime. 

        Return these parameters:
            frame_file_name, tags, len_box, box_ids, box_classes, box_vertices,
            len_poly, poly_ids, poly_classes, poly_vertices
        """
        frame_file_name, tags, box_ids, box_classes, box_vertices,\
            poly_ids, poly_classes, poly_vertices = cur_row 

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
    

    def _mask_bbox(self, mask_car, mask_rider, mask_ped, \
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
                tmp_img = Image.new('L', (self.width, self.height), 0)

                # Expand box((x0, y0), (x1, y1) to ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
                x0, y0, x1, y1 = (np.array(box_vertices[i])).flatten()

                x0 *= self.format_width_scale 
                x1 *= self.format_width_scale 
                y0 *= self.format_height_scale
                y1 *= self.format_height_scale

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


    def _mask_polygon(self, mask_car, mask_rider, mask_ped, \
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
                tmp_img = Image.new('L', (self.width, self.height), 0)

                scaled_list = [
                    p*self.format_width_scale  if i%2 else p*self.format_height_scale 
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


    def _gen_segmaps(self, cur_row):
        """
        Provided the CSV label record row, generate segmaps for labels. 

        Return imgaug SegmentationMapOnImage objects.
        """
        frame_file_name, tags, len_box, box_ids, box_classes, box_vertices,\
            len_poly, poly_ids, poly_classes, poly_vertices = self._parse_row(cur_row)
    
        # Create polygon masks
        mask_car = np.zeros((self.height, self.width))
        mask_rider = np.zeros((self.height, self.width))
        mask_ped = np.zeros((self.height, self.width))

        # Creating segmentation mask
    
        # Iterate through bbox classes and create mask using imgaug
        mask_car, mask_rider, mask_ped = self._mask_bbox(mask_car, mask_rider, mask_ped, \
                  len_box, box_ids, box_classes, box_vertices)

        # !!! NOTICE: latter mask will overwrite previous ones!!! 
        # Iterate through polygon classes and create mask using imgaug
        mask_car, mask_rider, mask_ped = self._mask_polygon(mask_car, mask_rider, mask_ped, \
                  len_poly, poly_ids, poly_classes, poly_vertices)

        # Convert to object representing a segmentation map associated with an image.
        #   Necessary for later augmentation.
        segmap_car = ia.SegmentationMapOnImage(np.uint8(mask_car), shape=(self.height, self.width),
                        nb_classes=1+np.int(np.max(mask_car)))
        segmap_rider = ia.SegmentationMapOnImage(np.uint8(mask_rider), shape=(self.height, self.width),
                        nb_classes=1+np.int(np.max(mask_rider)))
        segmap_ped = ia.SegmentationMapOnImage(np.uint8(mask_ped), shape=(self.height, self.width),
                        nb_classes=1+np.int(np.max(mask_ped)))

        return frame_file_name, segmap_car, segmap_rider, segmap_ped


    def exec_augment(self, cur_row):
        """
        Execute augmentation.
        """
        frame_file_name, segmap_car_obj, segmap_rider_obj, segmap_ped_obj = \
            self._gen_segmaps(cur_row)

        # Frame associated with the labels grouped
        frame_file_name = "".join([str(frame_file_name), self.format_ext])
        frame_file = "".join([self.export_path, "/", frame_file_name])
        print("frame_file:", frame_file)

         # Load an image (uint8) for later augmentation.
        self.image = cv2.imread(frame_file)

        # Augmentor
        self.seq_det = self.seq.to_deterministic()

        # Augment images and heatmaps.
        #   Return a list of imgaug object, but only one element, thus pick xxx[0].
        self.segmap_car_obj_aug = self.seq_det.augment_segmentation_maps([segmap_car_obj])[0]
        self.segmap_rider_obj_aug = self.seq_det.augment_segmentation_maps([segmap_rider_obj])[0]
        self.segmap_ped_obj_aug = self.seq_det.augment_segmentation_maps([segmap_ped_obj])[0]

        # Collect augmented segmaps
        #   draw() returns an RGB Image (H,W,3) ndarray(uint8)
        #   Only need one channel for mask generation. Reduce (H,W,3) -> (H,W) using np.any().
        tmp_segmap_aug = self.segmap_car_obj_aug.draw(size=(self.height, self.width))
        segmap_car_aug = np.any(tmp_segmap_aug, axis=2).astype(np.float)
        tmp_segmap_aug = self.segmap_rider_obj_aug.draw(size=(self.height, self.width))
        segmap_rider_aug = np.any(tmp_segmap_aug, axis=2).astype(np.float)
        tmp_segmap_aug = self.segmap_ped_obj_aug.draw(size=(self.height, self.width))
        segmap_ped_aug = np.any(tmp_segmap_aug, axis=2).astype(np.float)

        # Dilate on masks
        if self.dilation:
            segmap_car_aug_d = dilate_k_pix_ndimage(segmap_car_aug)
            segmap_rider_aug_d = dilate_k_pix_ndimage(segmap_rider_aug)
            segmap_ped_aug_d = dilate_k_pix_ndimage(segmap_ped_aug)
        else:
            segmap_car_aug_d = segmap_car_aug
            segmap_rider_aug_d = segmap_rider_aug
            segmap_ped_aug_d = segmap_ped_aug

        # Convert augmented segmap to augmented masks
        #   segmap_xxx_aug_d is in shape (272, 480), an array of True/False
        mask_car_aug_d = (segmap_car_aug_d).astype(np.float)
        mask_rider_aug_d = (segmap_rider_aug_d).astype(np.float)
        mask_ped_aug_d = (segmap_ped_aug_d).astype(np.float)

        return self.image, mask_car_aug_d, mask_rider_aug_d, mask_ped_aug_d


    def visualize(self):
        """
        Visualize the augmented image and masks.
        """
        # Pyplot needs color space conversion
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

        # Augment images
        image_aug = self.seq_det.augment_image(img)

        # Collect augmented segmaps on augmented images
        segmap_car_aug_on_image = self.segmap_car_obj_aug.draw_on_image(image_aug)
        segmap_rider_aug_on_image = self.segmap_rider_obj_aug.draw_on_image(image_aug)
        segmap_ped_aug_on_image = self.segmap_ped_obj_aug.draw_on_image(image_aug)

        return image_aug, segmap_car_aug_on_image, \
               segmap_rider_aug_on_image, segmap_ped_aug_on_image
