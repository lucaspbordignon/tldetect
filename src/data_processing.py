import pandas as pd
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
    labels = pd.concat(frames, ignore_index=True)
    labels = labels.drop_duplicates('file', keep='first')
    labels.reset_index(drop=True)
    return labels


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

    return X_train, X_val, X_test
