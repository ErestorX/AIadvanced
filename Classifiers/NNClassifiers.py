"""
Various classifiers based on the perceptron model each one with is own training and prediction process.

By Hugo LEMARCHANT
"""

from DataProcessing import IrisProcessing
import matplotlib.pyplot as plt
import scipy.special
import numpy as np


class NumpyNN:
    class Stats:
        def __init__(self, labels_dim):
            """
            Class used in NumpyNN to save information during the training session.
            :param labels_dim: dimension of the one hot label vector to generate the confusion matrix.
            """
            self.loss_times = []
            self.loss = []
            self.acc_times = []
            self.accuracy = []
            self.confusion = np.zeros((labels_dim, labels_dim))

        def reset(self, labels_dim):
            """
            Reset is called before each training session as this object is attached to it's NumpyNN object which could.
            be trained multiple times.
            :param labels_dim: dimension of the one hot label vector to generate the confusion matrix.
            :return:
            """
            self.loss_times = []
            self.loss = []
            self.acc_times = []
            self.accuracy = []
            self.confusion = np.zeros((labels_dim, labels_dim))

        def add_loss(self, time_index, new_loss):
            """
            This method is used each time a loss is calculated, it is stored withe the corresponding timestamp.
            :param time_index: integer corresponding to the timestamp of the new loss.
            :param new_loss: float which is the value of the calculated loss.
            :return:
            """
            self.loss_times.append(time_index)
            self.loss.append(new_loss)

        def add_accuracy(self, time_index, new_acc):
            """
            This method is used each time a accuracy score is calculated, it is stored withe the corresponding timestamp.
            :param time_index: integer corresponding to the timestamp of the new accuracy score.
            :param new_acc: float which is the value of the calculated accuracy score.
            :return:
            """
            self.accuracy.append(new_acc)
            self.acc_times.append(time_index)

        def print_loss_evol(self):
            """
            Prints the evolution of the loss during the training session.
            :return:
            """
            plt.title('Loss Evolution')
            plt.plot(self.loss_times, self.loss)

        def print_acc_evol(self):
            """
            Prints the evolution of the accuracy score during the training session.
            :return:
            """
            plt.title('Accuracy Evolution')
            plt.plot(self.acc_times, self.accuracy)

    def __init__(self, data_shape, hyp_param):
        """
        Instantiate a neural network with the given information about the dataset and its hyper-parameters.
        :param data_shape: information about the dataset (num_examples, nn_input_dim, nn_output_dim).
        :param hyp_param: chosen hyper parameters (nn_hdim, epsilon, reg_lambda).
        """
        np.random.seed(42)
        self.num_examples, self.nn_input_dim, self.nn_output_dim = data_shape
        self.nn_hdim, self.epsilon, self.reg_lambda = hyp_param
        self.W1 = np.random.randn(self.nn_input_dim, self.nn_hdim) / np.sqrt(self.nn_input_dim)
        self.b1 = np.zeros((1, self.nn_hdim))
        self.W2 = np.random.randn(self.nn_hdim, self.nn_output_dim) / np.sqrt(self.nn_hdim)
        self.b2 = np.zeros((1, self.nn_output_dim))
        self.stat = self.Stats(self.nn_output_dim)

    def train_model(self, features, labels, num_passes=2000):
        """
        This function learns parameters for the neural network and saves the model.
        :param features: numpy array containing the features.
        :param labels: numpy array containing the one hot labels.
        :param num_passes: integer, number of passes through the training data for gradient descent.
        :return:
        """
        self.stat.reset(self.nn_output_dim)
        for i in range(0, num_passes):
            # Forward propagation
            # Layer 1
            z1 = features.dot(self.W1) + self.b1
            # Activation 1
            a1 = np.tanh(z1)
            # Layer 2
            z2 = a1.dot(self.W2) + self.b2
            # Activation 2
            a2 = np.tanh(z2)
            # Softmax conversion
            exp_scores = scipy.special.expit(a2)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            # Backpropagation
            delta3 = probs
            accu = 0
            for j in range(self.num_examples):
                # For each prediction, calculate the error of the prediction
                index_labels = np.nonzero(labels[j])[0][0]
                index_probs = list(probs[j]).index(max(probs[j]))
                # Track the number of true positives
                if index_labels == index_probs:
                    accu += 1
                # If this is the last step of training, fill the confusion matrix
                if i == num_passes-1:
                    self.stat.confusion[index_labels][index_probs] += 1
                delta3[j, np.nonzero(labels[j])] -= 1
            dW2 = a1.T.dot(delta3) * 1/self.num_examples
            db2 = np.sum(delta3, axis=0, keepdims=True)
            delta2 = delta3.dot(self.W2.T) * (1 - np.power(a1, 2))
            dW1 = np.dot(features.T, delta2) * 1/self.num_examples
            db1 = np.sum(delta2, axis=0)
            # Add regularization terms (b1 and b2 don't have regularization terms)
            dW2 += self.reg_lambda * self.W2
            dW1 += self.reg_lambda * self.W1
            # Gradient descent parameter update
            self.W1 += -self.epsilon * dW1
            self.b1 += -self.epsilon * db1
            self.W2 += -self.epsilon * dW2
            self.b2 += -self.epsilon * db2
            self.stat.add_accuracy(i, accu/self.num_examples)
            # Optionally print the loss.
            # This is expensive because it uses the whole dataset, so we don't want to do it too often.
            if i % (num_passes/10) == 0:
                loss = self.calculate_loss(features, labels)
                self.stat.add_loss(i, loss)
                print("Iteration {0} : loss = {1}".format(i, loss))

    def calculate_loss(self, features, labels):
        """
        Calculate the loss explicitly in order to keep track of it on the stat class.
        :param features: numpy array containing the features.
        :param labels: numpy array containing the one hot labels.
        :return: float value of the mean square error regularized.
        """
        # Forward propagation to calculate our predictions
        z1 = features.dot(self.W1) + self.b1
        a1 = np.tanh(z1)
        z2 = a1.dot(self.W2) + self.b2
        a2 = np.tanh(z2)
        exp_scores = scipy.special.expit(a2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        # Calculating the loss
        corect_logprobs = []
        for i in range(self.num_examples) :
            corect_logprobs.append(-np.log(probs[i, np.nonzero(labels[i])]))
        data_loss = np.sum(corect_logprobs)
        # Add regularization term to loss
        data_loss += self.reg_lambda/2 * (np.sum(np.square(self.W1)) + np.sum(np.square(self.W2)))
        return 1./self.num_examples * data_loss

    def predict(self, X):
        """
        Predict classes given the current learned parameters.
        :param X: numpy array containing the features.
        :return: predictions: numpy array containing the prediction in one hot format.
        """
        z1 = X.dot(self.W1) + self.b1
        a1 = np.tanh(z1)
        z2 = a1.dot(self.W2) + self.b2
        a2 = np.tanh(z2)
        exp_scores = scipy.special.expit(a2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        predictions = []
        for j in range(len(X)):
            index_probs = list(probs[j]).index(max(probs[j]))
            predictions.append(index_probs)
        return np.array(predictions)

    def print_graph(self, accuracy=True, loss=True):
        """
        Wrapper to call Stats methods.
        :param accuracy: boolean to print or not accuracy graph.
        :param loss: boolean to print or not loss graph.
        :return:
        """
        if loss:
            self.stat.print_loss_evol()
        if accuracy:
            self.stat.print_acc_evol()
        plt.show()

    def print_confusion(self):
        """
        Wrapper to call Stats method.
        :return:
        """
        print('Confusion matrix :\n', self.stat.confusion)


def use_NumpyNN(dataset, epsilon=1e-3, reg_lambda=0.1, nn_hdim=256, num_passes=1000):
    """
    Function used to instantiate, train and test a numpy neural network.
    :param dataset: list of numpy arrays in the shape (training_features, training_labels, test_features, test_labels).
    :param epsilon: float, learning rate to slow down the gradient descent in order to converge.
    :param reg_lambda: float, regularization to take into count the current value of the weights during the
    gradient descent.
    :param nn_hdim: integer, number of neurons in the hidden layer.
    :param num_passes: integer, number of epochs of training.
    :return:
    """
    X, X_test, Y, Y_test = dataset
    onehot_Y = IrisProcessing.categorical_to_onehot(Y)
    train_dataset = (X, Y)
    data_shape = IrisProcessing.analyse_dataset(train_dataset, verbose=False)
    hyp_param = (nn_hdim, epsilon, reg_lambda)

    # build the model
    model = NumpyNN(data_shape, hyp_param)
    # train it
    model.train_model(X, onehot_Y, num_passes=num_passes)
    # print some information
    model.print_confusion()
    model.print_graph(loss=False)
    model.print_graph(accuracy=False)
    # use the trained model to predict
    H = model.predict(X_test)
    print("[INFO] predictions made : {0}.".format(H))


# TODO tensorflow/keras multi-layer NN
# TODO tensorflow/keras convolution NN
# TODO keras resnet


if __name__ == "__main__":
    use_NumpyNN(IrisProcessing.load_iris(0.1), epsilon=1e-4, nn_hdim=512, num_passes=2000)
