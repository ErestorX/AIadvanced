"""
Version 2.0
Code for extraction, standardization and analaysis of the Iris dataset.

By Hugo LEMARCHANT.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_iris(test_size):
    """
    Load_iris performs the loading of the Iris dataset present in the a folder named Dataset.
    It requires Iris dataset on .csv format.
    All features will be scaled to reduce scale bias between each feature.
    :param test_size: float in range [0,1[, wiche determine the portion of the dataset
                    to hold for the validation phase.
    :return
        X: the features which will be used as training data.
        X_test: the features used for validation phase.
        Y: the labels used for the training phase.
        Y_test: the labels usedd for the validation phase.
    :return:
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


def scale_dataset(*targets):
    """
    Scale_dataset scales each dataset with the sklearn function StandardScaler
    :param targets: a list of numpy arrays to scale.
    :return scaledDtatsets : a list of numpy arrays, each one scaled individualy.
    """
    print("[INFO] Scaling {0} sets...".format(len(targets)))
    scaler = StandardScaler()
    scaledDatasets = ()
    for set in targets:
        scaledDatasets = scaledDatasets + (scaler.fit_transform(set),)
    return scaledDatasets


def analyse_dataset(dataset, verbose=True):
    """
    Analyse_dataset prints several informations about the given dataset.
    It prints the number of exemples and features, as well as the number of diffrent classes.
    It shows a plot with a curve of the explained PCA variance and the representation of the dataset
    along the two bests components.
    :param dataset: a dataset wich is a tuple of numpy arrays (features, labels).
    :param verbose: boolean to print or not plots and [INFO] lines.
    :return: datashape : a list containing (nbExamples, nbFeatures, nbLabels).
    """
    X, Y = dataset
    label_list = np.unique(Y)
    if verbose:
        print("[INFO] dataset with {0} entries of {1} features.".format(len(Y), len(X[0])))
        print("[INFO] {0} labels : {1}.".format(len(label_list), label_list))
        plot_pca(X, Y)
        get_ratio_pca(X)
    return len(Y), len(X[0]), len(label_list)


def plot_pca(data, labels):
    """
    Plot_pca plots the representation of the datatset along the two bests components.
    :param data: numpy array of the features of the dataset to analyse.
    :param labels: numpy array of the labels of the corresponding dataset.
    :return:
    """
    # vector of all labels
    labels = np.array(labels)

    # pca with 2 components
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(data)
    principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])

    # plot pca
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 Component PCA', fontsize=20)

    finalDf = pd.concat([principalDf, pd.DataFrame(labels, columns=['target'])], axis=1)

    targets = np.unique(labels)
    colors = np.arange(len(np.unique(labels)))
    for target, color in zip(targets, colors):
        indicesToKeep = finalDf['target'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                   , finalDf.loc[indicesToKeep, 'principal component 2']
                   , s=50)
    ax.legend(targets)
    ax.grid()
    plt.show()


def get_ratio_pca(data):
    """
    Get_ratio_pca plots the energy cumulated of each component ordered.
    :param data: numpy array of the features of the dataset to analyse.
    :return:
    """
    pca = PCA().fit(data)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.title('Explained variance ratio for Iris dataset')
    plt.show()


def categorical_to_onehot(labels):
    """
    Generate the one hot representation of the given label list.
        The label list must be in the format: [C,A,D,B]
        The generate multi-labels representation will be:
        [
            [0 0 1 0],
            [1 0 0 0],
            [0 0 0 1],
            [0 1 0 0]
        ]
    :param labels: numpy array of the label list
    :return: numpy array of the generated multi-labels representation
    """
    label_map = []
    for label in labels:
        if label not in label_map:
            label_map.append(label)
    multi_lbls = []
    for label in labels:
        multi_lbls_elem = np.zeros(len(label_map))
        # Set a 1 at the value index
        value_indx = label_map.index(label)
        multi_lbls_elem[value_indx] = 1
        multi_lbls.append(multi_lbls_elem)
    return np.array(multi_lbls)


if __name__ == "__main__":
    X, X_test, Y, Y_test = load_iris(0.1)
    analyse_dataset((X, Y))
    onehotY = categorical_to_onehot(Y)
