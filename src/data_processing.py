from PIL import Image
import pandas as pd
import numpy as np
import random


DATA_DIR = 'data/object-dataset/'


def load_labels(filename='labels.csv',
                header=[
                    'file',
                    'x_min',
                    'y_min',
                    'x_max',
                    'y_max',
                    'occluded',
                    'label',
                    'properties']):
    """
        Loads a file and its labels. Extracts just the relevant data for the
        model, i.e. files with and without traffic lights.
    """

    raw_labels = pd.read_csv(DATA_DIR + filename, sep=' ', header=None,
                             names=header)

    tl_labels = raw_labels[raw_labels['label'] == 'trafficLight']
    tl_labels = tl_labels.drop_duplicates('file')

    non_tl_labels = raw_labels[raw_labels['label'] != 'trafficLight']
    non_tl_labels = non_tl_labels.drop_duplicates('file')

    frames = [tl_labels, non_tl_labels]
    all_labels = pd.concat(frames, ignore_index=True)
    all_labels = all_labels.drop_duplicates('file', keep='first')
    return all_labels.reset_index(drop=True)


def split_dataset(data, data_division={'train': 0.85, 'val': 0.15}):
    """
        Given a dataframe, splits it in training, validation and testing
        sets, with the data_divison policy.
    """

    if (data_division['train'] > 1):
        raise ValueError('Training data percentage must be between 0 and 1.')

    num_images = len(data)
    data_set = set(data)

    full_train_set = set(random.sample(data_set,
                         int(num_images * data_division['train'])))
    X_val = set(random.sample(full_train_set,
                int(len(full_train_set) * data_division['val'])))

    X_test = data_set.difference(full_train_set)
    X_train = full_train_set.difference(X_val)

    # Converting to numpy arrays
    X_train = np.array(list(X_train))
    X_val = np.array(list(X_val))
    X_test = np.array(list(X_test))

    return X_train, X_val, X_test


def extract_labels(dataframe, filenames):
    """
        Given a pandas dataframe, extract the labels and returns them as a
        numpy array.
    """
    dataframe = dataframe.set_index('file')
    labels = []
    for name in filenames:
        labels.append(dataframe.loc[name, 'label'])
    return np.array(labels)


def images_generator(X, y, batch_size):
    """
        A generator for the images. Loads a 'batch_size' of images form disk
        and returns it. Used for batch-training.

        As an example, if the batch size is 64:
        -    The first call will return data[0:64]
        -    The second call will return data[64:128]
        -    and so on.

        If the batch size is not a multiple of the dataset size, just the
        remaining data will be loaded at the last call.
    """
    count = 0
    total_images = X.shape[0]
    while True:
        begin_batch = count * batch_size
        end_batch = begin_batch + batch_size
        
        # Last chunk
        if end_batch > total_images:
            count = 0
            continue

        images = []
        labels = []

        for i in range(begin_batch, end_batch):
            images.append(np.array(Image.open('data/object-dataset/' + X[i])))
            if (y[i] == 'trafficLight'):
                labels.append(np.array([0, 1]))
            else:
                labels.append(np.array([1, 0]))
        count += 1
        yield (np.array(images), np.array(labels))
