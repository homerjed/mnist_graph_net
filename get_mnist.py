import array
import functools as ft
import gzip
import os
import struct
import urllib.request
import requests
import numpy as np 

import jax, jax.numpy as jnp


def load_mnist(train_data=True, test_data=False):
    """
    Get mnist data from the official website and
    load them in binary format.

    Parameters
    ----------
    train_data : bool
        Loads
        'train-images-idx3-ubyte.gz'
        'train-labels-idx1-ubyte.gz'
    test_data : bool
        Loads
        't10k-images-idx3-ubyte.gz'
        't10k-labels-idx1-ubyte.gz' 

    Return
    ------
    tuple
    tuple[0] are images (train & test)
    tuple[1] are labels (train & test)

    """
    RESOURCES = [
        'train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz']

    if (os.path.isdir('data') == 0):
        os.mkdir('data')
    if (os.path.isdir('data/mnist') == 0):
        os.mkdir('data/mnist')
    for name in RESOURCES:
        if (os.path.isfile('data/mnist/'+name) == 0):
            url = 'http://yann.lecun.com/exdb/mnist/'+name
            r = requests.get(url, allow_redirects=True)
            open('data/mnist/'+name, 'wb').write(r.content)

    return get_images(train_data, test_data), get_labels(train_data, test_data)


def get_images(train_data=True, test_data=False, binarize=False):

    to_return = []

    if train_data:
        with gzip.open('data/mnist/train-images-idx3-ubyte.gz', 'r') as f:
            # first 4 bytes is a magic number
            magic_number = int.from_bytes(f.read(4), 'big')
            # second 4 bytes is the number of images
            image_count = int.from_bytes(f.read(4), 'big')
            # third 4 bytes is the row count
            row_count = int.from_bytes(f.read(4), 'big')
            # fourth 4 bytes is the column count
            column_count = int.from_bytes(f.read(4), 'big')
            # rest is the image pixel data, each pixel is stored as an unsigned byte
            # pixel values are 0 to 255
            image_data = f.read()
            train_images = np.frombuffer(image_data, dtype=np.uint8)
            train_images = train_images.reshape((image_count, row_count, column_count))
            if binarize:
                to_return.append(np.where(train_images > 127, 1, 0))
            else:
                to_return.append(train_images)

    if test_data:
        with gzip.open('data/mnist/t10k-images-idx3-ubyte.gz', 'r') as f:
            # first 4 bytes is a magic number
            magic_number = int.from_bytes(f.read(4), 'big')
            # second 4 bytes is the number of images
            image_count = int.from_bytes(f.read(4), 'big')
            # third 4 bytes is the row count
            row_count = int.from_bytes(f.read(4), 'big')
            # fourth 4 bytes is the column count
            column_count = int.from_bytes(f.read(4), 'big')
            # rest is the image pixel data, each pixel is stored as an unsigned byte
            # pixel values are 0 to 255
            image_data = f.read()
            test_images = np.frombuffer(image_data, dtype=np.uint8)
            test_images = test_images.reshape((image_count, row_count, column_count))
            if binarize:
                to_return.append(np.where(test_images > 127, 1, 0))
            else:
                to_return.append(test_images)

    return jnp.asarray(to_return).reshape(-1, 28, 28, 1).astype(jnp.float32)


def get_labels(train_data=True, test_data=False):

    to_return = []

    if train_data:
        with gzip.open('data/mnist/train-labels-idx1-ubyte.gz', 'r') as f:
            # first 4 bytes is a magic number
            magic_number = int.from_bytes(f.read(4), 'big')
            # second 4 bytes is the number of labels
            label_count = int.from_bytes(f.read(4), 'big')
            # rest is the label data, each label is stored as unsigned byte
            # label values are 0 to 9
            label_data = f.read()
            train_labels = np.frombuffer(label_data, dtype=np.uint8)
            to_return.append(train_labels)
    if test_data:
        with gzip.open('data/mnist/t10k-labels-idx1-ubyte.gz', 'r') as f:
            # first 4 bytes is a magic number
            magic_number = int.from_bytes(f.read(4), 'big')
            # second 4 bytes is the number of labels
            label_count = int.from_bytes(f.read(4), 'big')
            # rest is the label data, each label is stored as unsigned byte
            # label values are 0 to 9
            label_data = f.read()
            test_labels = np.frombuffer(label_data, dtype=np.uint8)
            to_return.append(test_labels)

    labels = jnp.asarray(to_return, dtype=jnp.float32).reshape(-1, 1)
    return labels


if __name__ == "__main__":
    x, y = load_mnist()
    print(x.shape, y.shape)