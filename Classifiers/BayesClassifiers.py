"""
Various bayesian classifiers, which are probabilistic estimators.

By Hugo LEMARCHANT
"""

from DataProcessing import IrisProcessing
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


class sklearn_KNN():
    def __init__(self, dataset, max_neighbors=15):
        """
        Class that instantiate a KNN estimator. Its predictions are based on the mean class among the nearest neighbors.
        :param dataset: list of numpy arrays containing (training_features, test_features, training_labels, test_labels).
        :param max_neighbors: maximum amount of neighbors wanted to be explored.
        """
        self.best_uni_clf = KNeighborsClassifier(n_neighbors=1, weights='uniform')
        self.best_dist_clf = KNeighborsClassifier(n_neighbors=1, weights='distance')
        self.uniform_accuracy = []
        self.distance_accuracy = []
        # Assert that the max_neighbors is not higher than the number of examples available.
        if len(dataset[0]) < max_neighbors:
            self.max_neighbors = len(dataset[0])-1
        else:
            self.max_neighbors = max_neighbors
        # Explore each neighbors number classifier.
        for n_neighbor in range(1, max_neighbors):
            # Test both uniform and distance weights.
            # Distance weights weight the mean whereas uniform don't.
            uni_clf, uni_accu = self.train(dataset, n_neighbors=n_neighbor, weights='uniform')
            dist_clf, dist_accu = self.train(dataset, n_neighbors=n_neighbor, weights='distance')
            # Save each exploration result
            self.uniform_accuracy.append(uni_accu)
            self.distance_accuracy.append(dist_accu)
            # Save only the classifier if it is the best so far.
            if max(self.uniform_accuracy) == uni_accu:
                self.best_uni_clf = uni_clf
            if max(self.distance_accuracy) == dist_accu:
                self.best_dist_clf = uni_clf

    def train(self, dataset, n_neighbors=5, weights='uniform'):
        """
        Training of the KNN estimator.
        :param dataset: list of numpy arrays containing (training_features, test_features, training_labels, test_labels).
        :param n_neighbors: integer, number of neighbors to train with.
        :param weights: 'uniform' or 'distance', choice of the strategy to calculate the mean of the neighbors.
        :return: clf: the trained classifier.
                 accuracy: the accuracy of this classifier on the test set.
        """
        X, X_test, Y, Y_test = dataset
        clf = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
        clf = clf.fit(X, Y)
        H = clf.predict(X_test)
        accuracy = accuracy_score(Y_test, H)
        return clf, accuracy

    def predict(self, X, weights='uniform'):
        """
        Perform prediction of the given type of estimator on the X feature set.
        :param X: numpy array containing the features.
        :param weights: 'uniform' or 'distance', choice of the strategy to calculate the mean of the neighbors.
        :return: an array of the predictions.
        """
        predictions = []
        if weights == 'uniform':
            predictions = self.best_uni_clf.predict(X)
        if weights == 'distance':
            predictions = self.best_dist_clf.predict(X)
        return predictions

    def print_stats(self):
        """
        Print some information about the training process.
        :return:
        """
        for i in range(0, self.max_neighbors-1):
            plt.plot(i, self.uniform_accuracy[i], "o", color='red')
            plt.plot(i, self.distance_accuracy[i], "o", color='blue')
        plt.xlabel("n_neighbors")
        plt.ylabel("Accuracy")
        plt.title("Accuracy function of neighbors (red for uniform, blue for distance)")
        plt.xlim([0.5, self.max_neighbors-0.5])
        plt.ylim([0.75, 1])
        plt.show()


def use_sklearnKNN():
    """
    Basic function to instantiate, train and print some information of a KNN estimator.
    :return:
    """
    dataset = IrisProcessing.load_iris(0.15)
    model = sklearn_KNN(dataset)
    model.print_stats()


class sklearn_decisionTree():
    def __init__(self, dataset, max_depth=2):
        self.best_clf = DecisionTreeClassifier(max_depth=2)
        self.accuracies = []
        if max_depth >= len(dataset[0][0]):
            self.max_depth = len(dataset[0][0])-1
        else:
            self.max_depth = max_depth
        for depth in range(0, self.max_depth+1):
            if depth == 0:
                clf, accuracy = self.train(dataset, depth=None)
            else:
                clf, accuracy = self.train(dataset, depth=depth)
            self.accuracies.append(accuracy)
            if max(self.accuracies) == accuracy:
                self.best_clf = clf

    def train(self, dataset, depth=2):
        X, X_test, Y, Y_test = dataset
        clf = DecisionTreeClassifier(max_depth=depth)
        clf = clf.fit(X, Y)
        H = clf.predict(X_test)
        accuracy = accuracy_score(Y_test, H)
        return clf, accuracy

    def print_stats(self):
        for i in range(0, len(self.accuracies)):
            plt.plot(i, self.accuracies[i], "o")
        plt.xlabel("Depth")
        plt.ylabel("Accuracy")
        plt.title("Accuracy function of depth (0 meaning no max depth)")
        plt.xlim([-0.5, len(self.accuracies)-0.5])
        plt.ylim([0.0, 1])
        plt.show()


def use_sklearndecisionTree():
    """
    Basic function to instantiate, train and print some information of a decision tree estimator.
    :return:
    """
    dataset = IrisProcessing.load_iris(0.15)
    model = sklearn_decisionTree(dataset)
    model.print_stats()


class sklearn_naiveBayes():
    def __init__(self, dataset):
        self.NB_clf = [None, None]
        self.accuracies = [None, None]
        self.NB_clf[0], self.accuracies[0] = self.gaussian_train(dataset)
        self.NB_clf[1], self.accuracies[1] = self.multinomial_train(dataset)

    def multinomial_train(self, dataset):
        X, X_test, Y, Y_test = dataset
        est = KBinsDiscretizer(n_bins=10, encode='ordinal')
        est.fit(X)
        Xt = est.transform(X)
        est.fit(X_test)
        Xt_test = est.transform(X_test)

        clf = MultinomialNB()
        clf.fit(Xt, Y)
        H = clf.predict(Xt_test)

        accuracy = accuracy_score(Y_test, H)
        return clf, accuracy

    def gaussian_train(self, dataset):
        X, X_test, Y, Y_test = dataset
        clf = GaussianNB()
        clf.fit(X, Y)
        H = clf.predict(X_test)
        accuracy = accuracy_score(Y_test, H)
        return clf, accuracy


    def print_stats(self):
        print("[INFO] precisions for Gaussian and Multinomial naive bayesian classifiers:")
        print("[INFO] Gaussian accuracy : {}.".format(self.accuracies[0]))
        print("[INFO] Multinomial accuracy : {}.".format(self.accuracies[1]))


def use_sklearnnaiveBayes():
    """
    Basic function to instantiate, train and print some information of two naive bayes estimator.
    :return:
    """
    dataset = IrisProcessing.load_iris(0.15)
    model = sklearn_naiveBayes(dataset)
    model.print_stats()


if __name__ == "__main__":
    use_sklearnKNN()
    use_sklearndecisionTree()
    use_sklearnnaiveBayes()
