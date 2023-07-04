import pickle
import numpy as np
from config.train_config import SQRT_BATCH_SIZE
from tensorflow.data import Dataset


def getTrainDataset(root):
    dataset = []
    for i in range(5):
        with open(root + f'/data_batch_{i+1}', 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            dataset.append(np.reshape(dict[b'data'], (-1, 3, 32, 32)).transpose((0, 2, 3, 1)))
    dataset = np.vstack(dataset)

    dataset = dataset[0:int(dataset.shape[0]/SQRT_BATCH_SIZE**2)*SQRT_BATCH_SIZE**2]
    dataset = np.reshape(dataset, (-1, SQRT_BATCH_SIZE, SQRT_BATCH_SIZE, 32, 32, 3))
    dataset = np.transpose(dataset, (0, 1, 3, 2, 4, 5))
    dataset = np.reshape(dataset, (-1, 32*SQRT_BATCH_SIZE, 32*SQRT_BATCH_SIZE, 3))
    dataset = Dataset.from_tensor_slices(dataset)

    return dataset