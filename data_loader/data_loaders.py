
import pandas as pd

import os
from PIL import Image

import numpy as np

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
#from torch import np # Torch wrapper for Numpy
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from sklearn.preprocessing import MultiLabelBinarizer

from base import BaseDataLoader

from augmentation.dilate import *
from misc.lut import colorlut
from augmentation.aug import MyAugmentor


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super(MnistDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class MyDataset(Dataset):
    """
    Dataset wrapping images and target labels for data.

    Arguments:
        A CSV file path
        Path to image folder
        Extension of images
        PIL transforms
    """

    def __init__(self, csv_path, img_path, img_ext, training=False):
    
        tmp_df = pd.read_csv(csv_path)
        assert tmp_df['image_name'].apply(lambda x: os.path.isfile(img_path + x + img_ext)).all(), \
                "Some images referenced in the CSV file were not found"
        
        # Get rid of nan (empty tag) in 'tags' column
        tmp_df = tmp_df.fillna('none')
        self.tmp_df_val = tmp_df.values

        # TODO: How to load target from CVS with 3 classes instead of 4?
        #       Maybe just dropping the 2nd column. 
        #   The 4 classes are ordered as ['car', 'none', 'ped', 'rider'] alphabetically.
        #   E.g. 'car rider' will be translated into [1., 0., 0., 1.]
        self.mlb = MultiLabelBinarizer()
        self.img_path = img_path
        self.img_ext = img_ext
        # Normalize input
        self.training = training

        self.X_train = tmp_df['image_name']
        self.y_train = self.mlb.fit_transform(tmp_df['tags'].str.split()).astype(np.float32)

        self.myaug = MyAugmentor(training=self.training)

    def __getitem__(self, index):
        img, masks_aug_d, _ = self.myaug.exec_augment(self.tmp_df_val[index])
        # Old code as below:
        # img = Image.open(self.img_path + self.X_train[index] + self.img_ext)
        # img = img.convert('RGB')

        # Convert to float32 tensor
        img = transforms.functional.to_tensor(img).type(torch.float32)
        
        # TODO: label is redundant? augmenentation already returns label
        label = torch.from_numpy(self.y_train[index])

        return img, label

    def __len__(self):
        return len(self.X_train.index)

class MyDataLoader(BaseDataLoader):
    """
    Data loading using BaseDataLoader.
    """
    def __init__(self, data_dir, train_csv_file, test_csv_file, batch_size, shuffle, validation_split, num_workers, training=True):

        # TODO: make these configurable from .json config file
        IMG_PATH = data_dir + 'combined-jpg/'
        IMG_EXT = '.jpg'
        TRAIN_DATA = data_dir + train_csv_file
        # TODO: how to lead test dataset?
        TEST_DATA = data_dir + test_csv_file 

        self.dataset = MyDataset(TRAIN_DATA, IMG_PATH, IMG_EXT, training)
        super(MyDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
        
