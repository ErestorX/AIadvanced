"""
Code for extraction, standardization and analysis of the Iris dataset.

By Hugo LEMARCHANT.
"""

import pandas as pd
import numpy as np
from DataProcessing.datasetFunctions import scale_dataset
from sklearn.model_selection import train_test_split


def load_iris(test_size):
    """
    Load_iris performs the loading of the Iris dataset present in the a folder named Dataset.
    It requires Iris dataset on .csv format.
    All features will be scaled to reduce scale bias between each feature.
    :param test_size: float in range [0,1[, which determine the portion of the dataset
                    to hold for the validation phase.
    :return:
        X: the features which will be used as training data.
        X_test: the features used for validation phase.
        Y: the labels used for the training phase.
        Y_test: the labels used for the validation phase.
    """
    dataset = pd.read_csv('..\Datasets\iris.csv', delimiter=',')
    labels = dataset.values[:, -1:]
    features = dataset.values[:, :-1]
    print("[INFO] splitting the data ...")
    if test_size != 0:
        X, X_test, Y, Y_test = train_test_split(features, labels, test_size=test_size, random_state=42, stratify=labels)
    else:
        X, X_test, Y, Y_test = (features, [], labels, [])
    X, X_test = scale_dataset(X, X_test)
    return np.array(X), np.array(X_test), np.array(Y), np.array(Y_test)


if __name__ == "__main__":
    from DataProcessing.datasetFunctions import analyse_dataset, categorical_to_onehot
    X, X_test, Y, Y_test = load_iris(0.1)
    analyse_dataset((X, Y))
    onehotY = categorical_to_onehot(Y)
