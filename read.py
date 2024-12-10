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
import json

from PIL import Image
import json
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from tqdm import tqdm
import io
# def read_data(data_path):
#     for i in tqdm(range(len(os.listdir(data_path)))):
#         brand = os.listdir(data_path)[i]
#         img = imread(os.path.join(data_path, 'shot.png'))
#         img = img[:, :, 0:3] # RGB channels
#         all_imgs.append(resize(img, (reshape_size[0], reshape_size[1]), anti_aliasing=True))
#         all_labels.append(brand.split('+')[0])
#         all_file_names.append(os.path.join(data_path, brand, 'shot.png'))

#         all_imgs = np.asarray(all_imgs)
#         all_labels = np.asarray(all_labels)
#         return all_imgs, all_labels, all_file_names

# brand = os.listdir(r'C:\Users\вадим\VScode\VisualPhishNet\test')
# print(read_data(r'C:\Users\вадим\VScode\VisualPhishNet\test'))

# with open("test/shot.png", "rb") as image:
#   f = image.read()
#   b = bytearray(f)

#   data = {
#     "data": {
#         "img": str(b),
#         "url": "http://telegrarm.com/",
#         "html": "<div class=\"log\"> <a href=\"http://telegrarm.com\"> <img src=\"/static/temp320/picture/logotg.png\" data-rjs=\"2\" alt=""> </a> </div>"
#     }
# }

# with open('output.json', 'w') as file:
#     json.dump(data, file)

# with open("data_file.json", "r") as f:
#     data = f.read()
#     data_dict = json.loads(data)
    
#     img = data_dict["data"]["img"]
#     url = data_dict["data"]["url"]
#     html = data_dict["data"]["html"]

# for item in data_dict['data']:
#      print(item)

reshape_size = [224,224,3]

def read_data_from_json(json_path, reshape_size):
    '''
    read data from json file
    :param json_path: path to the json file containing bytearrays
    :param reshape_size: tuple for resizing images
    :return: images, labels, file_names
    '''
    all_imgs = []
    all_labels = []
    all_file_names = []

    with open(json_path, 'r') as f:
        data = f.read()
        data_dict = json.loads(data)
    
        img = data_dict["data"]["img"]
        url = data_dict["data"]["url"]
        html = data_dict["data"]["html"]
        
        byte_data = img  # Assuming the image is stored under the key 'image'
        
    for item in data_dict["data"]:
        brand = item       # Assuming the label is stored under the key 'label'
        
        # Convert bytearray to image
        img = Image.open(io.BytesIO(byte_data))
        img = np.array(img)[:, :, 0:3]  # RGB channels
        
        all_imgs.append(resize(img, (reshape_size[0], reshape_size[1]), anti_aliasing=True))
        all_labels.append(brand)
        all_file_names.append(item['file_name'])  # Assuming the file name is stored under 'file_name'

    all_imgs = np.asarray(all_imgs)
    all_labels = np.asarray(all_labels)
    return all_imgs, all_labels, all_file_names

read_data_from_json('data_file.json', reshape_size)
