"""
Code for standardization and analysis of a dataset.

By Hugo LEMARCHANT.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def scale_dataset(*targets):
    """
    Scale_dataset scales each dataset with the sklearn function StandardScaler
    :param targets: a list of numpy arrays to scale.
    :return scaledDtatsets : a list of numpy arrays, each one scaled individually.
    """
    print("[INFO] Scaling {0} sets...".format(len(targets)))
    scaler = StandardScaler()
    scaledDatasets = ()
    for set in targets:
        scaledDatasets = scaledDatasets + (scaler.fit_transform(set),)
    return scaledDatasets


def analyse_dataset(dataset, one_hot=True):
    """
    Analyse_dataset prints several information about the given dataset.
    It prints the number of examples and features, as well as the number of different classes.
    It shows a plot with a curve of the explained PCA variance and the representation of the dataset
    along the two bests components.
    :param dataset: a dataset which is a tuple of numpy arrays (features, labels).
    :param one_hot: boolean to perform functions compatible with this label format.
    :return: datashape : a list containing (nbExamples, nbFeatures, nbLabels).
    """
    X, Y = dataset
    if one_hot:
        label_list = np.unique(Y, axis=0)
    else:
        label_list = np.unique(Y)
    print("[INFO] dataset with {0} entries of {1} features.".format(len(Y), len(X[0])))
    print("[INFO] {0} labels : {1} of type {2}.".format(len(label_list), label_list, type(label_list[0])))
    if not one_hot:
        get_ratio_pca(X)
        plot_pca(X, Y)
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
    Get_ratio_pca plots the cumulative energy of each component ordered.
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
