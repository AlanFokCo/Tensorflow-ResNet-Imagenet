import os
import tensorflow as tf
import numpy as np


def get_img_index(path):
    path_dic = {}
    for root, dirs, _ in os.walk(path, topdown=False):

        for name in dirs:
            temp_arr = os.listdir(os.path.join(root, name))
            for idx in range(0, len(temp_arr)):
                temp_arr[idx] = os.path.join(root, name, temp_arr[idx])
            path_dic[name] = temp_arr

    return path_dic


def load_data_with_split(para=1,
                         index=0,
                         dataset=str,
                         shape=None):
    if shape is None:
        shape = [224, 224]
    path_dic = get_img_index(dataset)
    img_sum = 0

    file_arr = []
    labels = []
    idx = 0

    for key in path_dic:
        img_sum += len(path_dic[key])
        file_arr.extend(path_dic[key])
        labels.extend([idx] * len(path_dic[key]))
        idx += 1

    per_num = (img_sum // para) + 1
    start_index = index * per_num
    end_index = (index + 1) * per_num

    images = file_arr[start_index:end_index]
    data = []

    for image in images:
        data.append(decode_img(image, shape))

    data = np.array(data)
    labels = np.array(labels)

    return data, labels


def load_data_with_split_only_index(para=1,
                                    index=0,
                                    dataset=str,
                                    shape=None):
    if shape is None:
        shape = [224, 224]
    path_dic = get_img_index(dataset)
    img_sum = 0

    file_arr = []
    labels = []
    idx = 0

    for key in path_dic:
        img_sum += len(path_dic[key])
        file_arr.extend(path_dic[key])
        labels.extend([idx] * len(path_dic[key]))
        idx += 1

    per_num = (img_sum // para) + 1
    start_index = index * per_num
    end_index = (index + 1) * per_num

    file_arr = file_arr[start_index:end_index]

    labels = np.array(labels)

    return file_arr, labels


def decode_img(img_path, shape=None):
    if shape is None:
        shape = [224, 224]
    image = tf.io.read_file(img_path)
    image = tf.io.decode_jpeg(image)

    image = tf.image.resize(image, shape)
    image = image / 255.

    if image.shape != (224, 224, 3):
        image = np.random.random(size=(224, 224, 3))

    return image

