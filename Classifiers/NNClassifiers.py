"""
Various classifiers based on the perceptron model each one with is own training and prediction process.

By Hugo LEMARCHANT
"""

import DataProcessing.datasetFunctions as df
import matplotlib.pyplot as plt
import tensorflow as tf
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

        def fill_confusion(self, index_labels, index_probs):
            self.confusion[index_labels][index_probs] += 1

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

    def forward_prop(self, features):
        """
        Passes a given set of examples through the network and returns the activation or useful back_prop information.
        :param features: numpy array containing each the values of each example.
        :return:
            a1: numpy array which is the activation of the hidden layer.
            h: numpy array containing the activations of the output layer.
        """
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
        h = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return a1, h

    def back_prop(self, num_epochs, current_epoch, features, labels, a1, h):
        """
        Perform stochastic gradient descent and update the weights.
        :param num_epochs: integer number of epochs to train our network.
        :param current_epoch: integer indicating the current epoch of the training.
        :param features: numpy array containing each the values of each example.
        :param labels: numpy array containing each label of each example on one hot format.
        :param a1: numpy array which is the activation of the hidden layer.
        :param h: numpy array containing the activations of the output layer.
        :return:
        """
        # Backpropagation
        accu = 0
        for j in range(self.num_examples):
            # For each prediction, calculate the error of the prediction
            index_labels = np.nonzero(labels[j])[0][0]
            index_probs = list(h[j]).index(max(h[j]))
            # Track the number of true positives
            if index_labels == index_probs:
                accu += 1
            # If this is the last step of training, fill the confusion matrix
            if current_epoch == num_epochs - 1:
                self.stat.fill_confusion(index_labels, index_probs)
            h[j, np.nonzero(labels[j])[0][0]] -= 1
        self.stat.add_accuracy(current_epoch, accu / self.num_examples)
        dW2 = a1.T.dot(h) * 1 / self.num_examples
        db2 = np.sum(h, axis=0, keepdims=True)
        delta2 = h.dot(self.W2.T) * (1 - np.power(a1, 2))
        dW1 = features.T.dot(delta2) * 1 / self.num_examples
        db1 = np.sum(delta2, axis=0)
        # Add regularization terms (b1 and b2 don't have regularization terms)
        dW2 += self.reg_lambda * self.W2
        dW1 += self.reg_lambda * self.W1
        # Gradient descent parameter update
        self.W1 += -self.epsilon * dW1
        self.b1 += -self.epsilon * db1
        self.W2 += -self.epsilon * dW2
        self.b2 += -self.epsilon * db2

    def train_model(self, features, labels, num_epochs=2000):
        """
        Function to iterate through each epoch and perform the learning process.
        :param features: numpy array containing each the values of each example.
        :param labels: numpy array containing each label of each example on one hot format.
        :param num_epochs: integer number of epochs to train our network.
        :return:
        """
        self.stat.reset(self.nn_output_dim)
        for i in range(0, num_epochs):
            a1, h = self.forward_prop(features)
            # Optionally print the loss.
            if i % (num_epochs / 10) == 0:
                self.calculate_loss(i, h, labels)
            self.back_prop(num_epochs, i, features, labels, a1, h)

    def calculate_loss(self, epoch, h, labels):
        """
        Calculate the loss explicitly in order to keep track of it on the stat class.
        :param epoch: integer the current epoch.
        :param h: numpy array containing the predictions.
        :param labels: numpy array containing the one hot labels.
        :return: float value of the mean square error regularized.
        """
        corect_logprobs = []
        for i in range(self.num_examples):
            corect_logprobs.append(-np.log(h[i, np.nonzero(labels[i])]))
        data_loss = np.sum(corect_logprobs)
        # Add regularization term to loss
        data_loss += self.reg_lambda/2 * (np.sum(np.square(self.W1)) + np.sum(np.square(self.W2)))
        loss = 1./self.num_examples * data_loss
        self.stat.add_loss(epoch, loss)
        print("Iteration {0} : loss = {1}".format(epoch, loss))
        return loss

    def predict(self, X):
        """
        Predict classes given the current learned parameters.
        :param X: numpy array containing the features.
        :return: predictions: numpy array containing the prediction in one hot format.
        """
        a1, H = self.forward_prop(X)
        index_probs = []
        for prediction in H:
            index_probs.append(list(prediction).index(max(prediction)))
        return index_probs

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


def use_NumpyNN(dataset, epsilon=1e-3, reg_lambda=0.1, nn_hdim=256, num_passes=2000):
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
    train_dataset = (X, Y)
    data_shape = df.analyse_dataset(train_dataset)
    hyp_param = (nn_hdim, epsilon, reg_lambda)

    # build the model
    model = NumpyNN(data_shape, hyp_param)
    # train it
    model.train_model(X, Y, num_epochs=num_passes)
    # print some information
    model.print_confusion()
    model.print_graph(loss=False)
    model.print_graph(accuracy=False)
    # use the trained model to predict
    H = model.predict(X_test)
    print("[INFO] predictions made : {0}.".format(H))


class TfNN:
    def __init__(self):
        RANDOM_SEED = 42
        tf.set_random_seed(RANDOM_SEED)

    def init_weights(shape):
        """ Weight initialization """
        weights = tf.random_normal(shape, stddev=0.1)
        return tf.Variable(weights)

    def forwardprop(X, w_1, w_2):
        """
        Forward-propagation.
        IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
        """
        h = tf.nn.sigmoid(tf.matmul(X, w_1))  # The \sigma function
        yhat = tf.matmul(h, w_2)  # The \varphi function
        return yhat

    def get_iris_data(self):
        """ Read the iris data set and split them into training and test sets """
        iris = datasets.load_iris()
        data = iris["data"]
        target = iris["target"]

        # Prepend the column of 1s for bias
        N, M = data.shape
        all_X = np.ones((N, M + 1))
        all_X[:, 1:] = data

        # Convert into one-hot vectors
        num_labels = len(np.unique(target))
        all_Y = np.eye(num_labels)[target]  # One liner trick!
        return train_test_split(all_X, all_Y, test_size=0.33, random_state=RANDOM_SEED)

    def train(self):
        train_X, test_X, train_y, test_y = get_iris_data()

        # Layer's sizes
        x_size = train_X.shape[1]  # Number of input nodes: 4 features and 1 bias
        h_size = 256  # Number of hidden nodes
        y_size = train_y.shape[1]  # Number of outcomes (3 iris flowers)

        # Symbols
        X = tf.placeholder("float", shape=[None, x_size])
        y = tf.placeholder("float", shape=[None, y_size])

        # Weight initializations
        w_1 = init_weights((x_size, h_size))
        w_2 = init_weights((h_size, y_size))

        # Forward propagation
        yhat = forwardprop(X, w_1, w_2)
        predict = tf.argmax(yhat, axis=1)

        # Backward propagation
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
        updates = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

        # Run SGD
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)

        for epoch in range(100):
            # Train with each example
            for i in range(len(train_X)):
                sess.run(updates, feed_dict={X: train_X[i: i + 1], y: train_y[i: i + 1]})

            train_accuracy = np.mean(np.argmax(train_y, axis=1) ==
                                     sess.run(predict, feed_dict={X: train_X, y: train_y}))
            test_accuracy = np.mean(np.argmax(test_y, axis=1) ==
                                    sess.run(predict, feed_dict={X: test_X, y: test_y}))

            print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%"
                  % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy))

        sess.close()


# TODO tensorflow/keras convolution NN
# TODO keras resnet


if __name__ == "__main__":
    from DataProcessing import IrisProcessing, mnistProcessing
    use_NumpyNN(IrisProcessing.load_iris(0.1), epsilon=1e-5, nn_hdim=1024, num_passes=5000)
    use_NumpyNN(mnistProcessing.load_MNIST(flatten=True, one_hot=True), epsilon=1e-5, nn_hdim=256, num_passes=500)
