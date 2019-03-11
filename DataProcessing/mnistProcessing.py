"""
Code for extraction of the MNIST and FashionMNIST datasets.

By Hugo LEMARCHANT.
"""

from tensorflow.contrib.learn.python.learn.datasets.mnist import extract_images, extract_labels
from DataProcessing.datasetFunctions import scale_dataset, categorical_to_onehot


def load_FashionMNIST(flatten=False, one_hot=False):
    """
    load_FashionMNIST performs the loading of the FashionMNIST dataset present in the a folder named Dataset\FashionMNIST.
    It requires FashionMNIST dataset on bytecode format.
    All features will be scaled to reduce scale bias between each feature.
    :param flatten: boolean to flatten the dataset for fully-connected NN.
    :param one_hot: boolean to convert integer labels into one hot format.
    :return:
    """
    with open('../Datasets/FashionMNIST/train-images-idx3-ubyte.gz', 'rb') as f:
        X = extract_images(f)
    with open('../Datasets/FashionMNIST/train-labels-idx1-ubyte.gz', 'rb') as f:
        Y = extract_labels(f)

    with open('../Datasets/FashionMNIST/t10k-images-idx3-ubyte.gz', 'rb') as f:
        X_test = extract_images(f)
    with open('../Datasets/FashionMNIST/t10k-labels-idx1-ubyte.gz', 'rb') as f:
        Y_test = extract_labels(f)

    if flatten:
        X = X.flatten().reshape(60000, 784)
        X_test = X_test.flatten().reshape(10000, 784)
        X, X_test = scale_dataset(X, X_test)
    if one_hot:
        Y, Y_test = categorical_to_onehot(Y), categorical_to_onehot(Y_test)
    return X, X_test, Y, Y_test


def load_MNIST(flatten=False, one_hot=False):
    """
    load_MNIST performs the loading of the MNIST dataset present in the a folder named Dataset\MNIST.
    It requires MNIST dataset on bytecode format.
    All features will be scaled to reduce scale bias between each feature.
    :param flatten: boolean to flatten the dataset for fully-connected NN.
    :param one_hot: boolean to convert integer labels into one hot format.
    :return:
    """
    with open('../Datasets/MNIST/train-images-idx3-ubyte.gz', 'rb') as f:
        X = extract_images(f)
    with open('../Datasets/MNIST/train-labels-idx1-ubyte.gz', 'rb') as f:
        Y = extract_labels(f)

    with open('../Datasets/MNIST/t10k-images-idx3-ubyte.gz', 'rb') as f:
        X_test = extract_images(f)
    with open('../Datasets/MNIST/t10k-labels-idx1-ubyte.gz', 'rb') as f:
        Y_test = extract_labels(f)

    if flatten:
        X = X.flatten().reshape(60000, 784)
        X_test = X_test.flatten().reshape(10000, 784)
        X, X_test = scale_dataset(X, X_test)
    if one_hot:
        Y, Y_test = categorical_to_onehot(Y), categorical_to_onehot(Y_test)
    return X, X_test, Y, Y_test


if __name__=="__main__":
    from DataProcessing.datasetFunctions import analyse_dataset
    X, X_test, Y, Y_test = load_MNIST(flatten=True)
    analyse_dataset((X, Y), one_hot=False)
    X, X_test, Y, Y_test = load_FashionMNIST(flatten=True)
    analyse_dataset((X, Y), one_hot=False)
