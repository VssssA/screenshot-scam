import numpy as np
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten,Subtract,Reshape
from keras.preprocessing import image
from keras.models import Model
from keras.regularizers import l2
from keras import optimizers

from skimage.io import imsave

import os
from tqdm import tqdm

from visualphish_model import *
import math

def read_data(data_path):
    for i in tqdm(range(len(os.listdir(data_path)))):
        brand = os.listdir(data_path)[i]
        img = imread(os.path.join(data_path, 'shot.png'))
        img = img[:, :, 0:3] # RGB channels
        all_imgs.append(resize(img, (reshape_size[0], reshape_size[1]), anti_aliasing=True))
        all_labels.append(brand.split('+')[0])
        all_file_names.append(os.path.join(data_path, brand, 'shot.png'))

        all_imgs = np.asarray(all_imgs)
        all_labels = np.asarray(all_labels)
        return all_imgs, all_labels, all_file_names

# brand = os.listdir(r'C:\Users\вадим\VScode\VisualPhishNet\test')
print(read_data(r'C:\Users\вадим\VScode\VisualPhishNet\test'))
