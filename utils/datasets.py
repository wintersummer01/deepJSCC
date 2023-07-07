import pickle
import numpy as np
import matplotlib.pyplot as plt
from config.train_config import SQRT_BATCH_SIZE
from config.train_config import DATA_TYPE
import tensorflow as tf


def getTrainDataset(root, show_spec=False):
    dataset = []
    for i in range(5):
        with open(root + f'/data_batch_{i+1}', 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            dataset.append(np.reshape(dict[b'data'], (-1, 3, 32, 32)).transpose((0, 2, 3, 1)))
    dataset = np.vstack(dataset)

    if DATA_TYPE == 'CIFAR-10':
        dataset = dataset[0:int(dataset.shape[0]/SQRT_BATCH_SIZE**2)*SQRT_BATCH_SIZE**2]
        dataset = np.reshape(dataset, (-1, SQRT_BATCH_SIZE, SQRT_BATCH_SIZE, 32, 32, 3))
        dataset = np.transpose(dataset, (0, 1, 3, 2, 4, 5))
        dataset = np.reshape(dataset, (-1, 1, 32*SQRT_BATCH_SIZE, 32*SQRT_BATCH_SIZE, 3))
        
    if show_spec:
        print('length of dataset: ', len(dataset))
        print('dataset image shape:', dataset[0][0].shape)
        print('example image:')
        plt.imshow(dataset[0][0])
        plt.show()
    
    return tf.data.Dataset.from_tensor_slices(tf.cast(dataset, tf.float32)), dataset[0][0].shape


def getTestDataset(root, show_spec=False):
    with open(root + f'/test_batch', 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        dataset = np.reshape(dict[b'data'], (-1, 3, 32, 32)).transpose((0, 2, 3, 1))

    if DATA_TYPE == 'CIFAR-10':
        dataset = dataset[0:int(dataset.shape[0]/SQRT_BATCH_SIZE**2)*SQRT_BATCH_SIZE**2]
        dataset = np.reshape(dataset, (-1, SQRT_BATCH_SIZE, SQRT_BATCH_SIZE, 32, 32, 3))
        dataset = np.transpose(dataset, (0, 1, 3, 2, 4, 5))
        dataset = np.reshape(dataset, (-1, 1, 32*SQRT_BATCH_SIZE, 32*SQRT_BATCH_SIZE, 3))
        
    if show_spec:
        print('length of dataset: ', len(dataset))
        print('dataset image shape:', dataset[0][0].shape)
        print('example image:')
        plt.imshow(dataset[0][0])
        plt.show()
    
    return tf.data.Dataset.from_tensor_slices(tf.cast(dataset, tf.float32)), dataset[0][0].shape