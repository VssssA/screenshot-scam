
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

import os
import io
import PIL.Image as Image
import cv2
from array import array

# Data
reshape_size = [224,224,3]
model_path = 'my_model2.h5'
targetlist_emb_path = 'targetlist_emb.npy'
targetlist_label_path = 'targetlist_labels.npy'
targetlist_file_name_path = 'targetlist_filenames.npy'

# load model
def load_weights(model_path):
    '''
    Load model
    :param model_path:
    :return: final_model: complete model which returns class
    :return: inside_model: partial model which returns an intermediate embedding
    '''
    margin = 2.2
    input_shape = [224, 224, 3]
    new_conv_params = [3, 3, 512]

    final_model = define_triplet_network(input_shape, new_conv_params)
    final_model.summary()

    final_model.load_weights(model_path)
    inside_model = final_model.layers[3]  # partial model to get the embedding
    return final_model, inside_model

# load targetlist embedding
def load_targetemb(emb_path, label_path, file_name_path):
    '''
    load targetlist embedding
    :return:
    '''
    targetlist_emb = np.load(emb_path)
    all_labels = np.load(label_path)
    all_file_names = np.load(file_name_path)
    return targetlist_emb, all_labels, all_file_names



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
        json_data = json.load(f)
        
        data = json_data['data']
        
    for item in tqdm(data):
        byte_data = item['img']  # Assuming the image is stored under the key 'image'
        brand = item['label']       # Assuming the label is stored under the key 'label'
        
        # Convert bytearray to image
        img = Image.open(io.BytesIO(byte_data))
        img = np.array(img)[:, :, 0:3]  # RGB channels
        
        all_imgs.append(resize(img, (reshape_size[0], reshape_size[1]), anti_aliasing=True))
        all_labels.append(brand)
        all_file_names.append(item['file_name'])  # Assuming the file name is stored under 'file_name'

    all_imgs = np.asarray(all_imgs)
    all_labels = np.asarray(all_labels)

    return all_imgs, all_labels, all_file_names


# L2 distance
def compute_distance_pair(layer1, layer2, targetlist_emb):
    diff = layer1 - layer2
    l2_diff = np.sum(diff ** 2) / targetlist_emb.shape[1]
    return l2_diff


# Pairwise distance between query image and training
def compute_all_distances(test_matrix, targetlist_emb):
    train_size = targetlist_emb.shape[0]
    pairwise_distance = np.zeros([test_matrix.shape[0], train_size])

    for i in tqdm(range(test_matrix.shape[0])):  # every instance in test_matrix
        pair1 = test_matrix[i, :]
        for j in range(0, train_size):
            pair2 = targetlist_emb[j, :]
            l2_diff = compute_distance_pair(pair1, pair2, targetlist_emb)
            pairwise_distance[i, j] = l2_diff

    return pairwise_distance


# Find Smallest n distances
def find_min_distances(distances, n):
    idx = distances.argsort()[:n]
    values = distances[idx]
    return idx, values


# Find names of examples with min distance
def find_names_min_distances(idx, values, all_file_names):
    names_min_distance = ''
    only_names = []
    distances = ''
    for i in range(idx.shape[0]):
        index_min_distance = idx[i]
        names_min_distance = names_min_distance + 'Targetlist: ' + all_file_names[index_min_distance] + ','
        only_names.append(all_file_names[index_min_distance])
        distances = distances + str(values[i]) + ','

    names_min_distance = names_min_distance[:-1]
    distances = distances[:-1]

    return names_min_distance, only_names, distances

def read_data(data_path, reshape_size):
    '''
    read data
    :param data_path:
    :param reshape_size:
    :param chunk_range: Tuple
    :return:
    '''
    all_imgs = []
    all_labels = []
    all_file_names = []

    for i in tqdm(range(len(os.listdir(data_path)))):
        brand = os.listdir(data_path)[i]
        img = imread(os.path.join(data_path, 'shot.png'))
        img = img[:, :, 0:3] # RGB channels
        all_imgs.append(resize(img, (reshape_size[0], reshape_size[1]), anti_aliasing=True))
        # reshaped_img = resize(img, (reshape_size[0], reshape_size[1]), anti_aliasing=True)
        # reshaped_img.save('result4.png')
        all_labels.append(brand.split('+')[0])
        all_file_names.append(os.path.join(data_path, brand))

    all_imgs = np.asarray(all_imgs)
    all_labels = np.asarray(all_labels)
    return all_imgs, all_labels, all_file_names


def result_without_dir(img_bytes, ts = 1.5):

    all_images = []
    targetlist_emb, all_labels, all_file_names = load_targetemb(targetlist_emb_path, targetlist_label_path, targetlist_file_name_path)
    all_file_names = [x.split('../')[1] for x in all_file_names]
    _, inside_model = load_weights(model_path)
    print('Loaded targetlist and model, number of protected target screenshots {}'.format(len(targetlist_emb)))
    
    # image = Image.open(io.BytesIO(img_bytes))
    # image.save('saved.png')
    nparr = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    cv2.imwrite('result.png',image)
    image = imread('result.png')
    image = image[:, :, 0:3]
    # cv2.imwrite('result2.png',image)
    all_images.append(resize(image, (reshape_size[0], reshape_size[1]), anti_aliasing=True))
    # reshaped_img = resize(image, (reshape_size[0], reshape_size[1]), anti_aliasing=True)
    # cv2.imwrite('result3.png',reshaped_img)
    all_images = np.asarray(all_images)
    data_emb = inside_model.predict(all_images, batch_size=32)
    pairwise_distance = compute_all_distances(data_emb, targetlist_emb)
    # print(pairwise_distance)
    n=1
    print('Start ')
    for i in tqdm(range(data_emb.shape[0])):
        distances_to_target = pairwise_distance[i,:]
        idx, values = find_min_distances(np.ravel(distances_to_target), n)
        names_min_distance, only_names, min_distances = find_names_min_distances(idx, values, all_file_names)

        # distance lower than threshold ==> report as phishing
        if float(min_distances) <= ts:
            phish = "Phishing"
        # else it is benign
        else:
            phish = "Not Phishing"
        print(str(phish)+str(min_distances)+str(only_names[0]))


def main(data_folder, result_path, ts = 1.5):
    # load targetlist and model
    targetlist_emb, all_labels, all_file_names = load_targetemb(targetlist_emb_path, targetlist_label_path, targetlist_file_name_path)
    all_file_names = [x.split('../')[1] for x in all_file_names]
    _, inside_model = load_weights(model_path)
    print('Loaded targetlist and model, number of protected target screenshots {}'.format(len(targetlist_emb)))
        

    # read data
    X, y, file_names = read_data(data_folder, reshape_size)
    print('Finish reading data, number of data {}'.format(len(X)))

    # get embeddings from data
    data_emb = inside_model.predict(X, batch_size=32)
    pairwise_distance = compute_all_distances(data_emb, targetlist_emb)
    print('Finish getting embedding')

    n = 1 #Top-1 match
    print('Start ')
    for i in tqdm(range(data_emb.shape[0])):
        url = open(file_names[i].replace('shot.png', 'info.txt'), encoding='utf-8', errors='ignore').read()
        print(url)
        distances_to_target = pairwise_distance[i,:]
        idx, values = find_min_distances(np.ravel(distances_to_target), n)
        names_min_distance, only_names, min_distances = find_names_min_distances(idx, values, all_file_names)

        # distance lower than threshold ==> report as phishing
        if float(min_distances) <= ts:
            phish = "Phishing"
        # else it is benign
        else:
            phish = "Not Phishing"
        # print(str(phish))

        print(url+'\t'+str(phish)+'\t'+str(min_distances)+str(only_names[0])+'\n')
        # with open(result_path, 'a+', encoding='utf-8', errors='ignore') as f:
        #     f.write(url+'\t'+str(phish)+'\t'+str(min_distances)+str(only_names[0])+'\n')






