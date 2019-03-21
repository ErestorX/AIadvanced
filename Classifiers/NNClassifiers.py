"""
Various classifiers based on the perceptron model each one with is own training and prediction process.

By Hugo LEMARCHANT
"""
import DataProcessing.datasetFunctions as df
import DataProcessing.mnistProcessing
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.special
import numpy as np
from tensorflow.python.framework import ops
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.keras import backend
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras import regularizers


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


def use_NumpyNN(dataset, epsilon=1e-5, reg_lambda=0.1, nn_hdim=256, num_passes=2000):
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
    """
    Code from "https://medium.com/tensorist/classifying-fashion-articles-using-tensorflow-fashion-mnist-f22e8a04728a"
    :return:
    """
    def __init__(self, mnist):
        # Network parameters
        self.n_hidden_1 = 128  # Units in first hidden layer
        self.n_hidden_2 = 128  # Units in second hidden layer
        self.n_input = 784  # Fashion MNIST data input (img shape: 28*28)
        self.n_classes = 10  # Fashion MNIST total classes (0–9 digits)
        self.n_samples = mnist.train.num_examples  # Number of examples in training set

    def create_placeholders(self, n_x, n_y):
        """
        Creates the placeholders for the tensorflow session.

        Arguments:
        n_x -- scalar, size of an image vector (28*28 = 784)
        n_y -- scalar, number of classes (10)

        Returns:
        X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
        Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"
        """

        X = tf.placeholder(tf.float32, [n_x, None], name="X")
        Y = tf.placeholder(tf.float32, [n_y, None], name="Y")

        return X, Y

    def initialize_parameters(self):
        """
        Initializes parameters to build a neural network with tensorflow. The shapes are:
                            W1 : [n_hidden_1, n_input]
                            b1 : [n_hidden_1, 1]
                            W2 : [n_hidden_2, n_hidden_1]
                            b2 : [n_hidden_2, 1]
                            W3 : [n_classes, n_hidden_2]
                            b3 : [n_classes, 1]

        Returns:
        parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
        """

        # Set random seed for reproducibility
        tf.set_random_seed(42)

        # Initialize weights and biases for each layer
        # First hidden layer
        W1 = tf.get_variable("W1", [self.n_hidden_1, self.n_input], initializer=tf.contrib.layers.xavier_initializer(seed=42))
        b1 = tf.get_variable("b1", [self.n_hidden_1, 1], initializer=tf.zeros_initializer())

        # Second hidden layer
        W2 = tf.get_variable("W2", [self.n_hidden_2, self.n_hidden_1], initializer=tf.contrib.layers.xavier_initializer(seed=42))
        b2 = tf.get_variable("b2", [self.n_hidden_2, 1], initializer=tf.zeros_initializer())

        # Output layer
        W3 = tf.get_variable("W3", [self.n_classes, self.n_hidden_2], initializer=tf.contrib.layers.xavier_initializer(seed=42))
        b3 = tf.get_variable("b3", [self.n_classes, 1], initializer=tf.zeros_initializer())

        # Store initializations as a dictionary of parameters
        parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}

        return parameters

    def forward_propagation(self, X, parameters):
        """
        Implements the forward propagation for the model:
        LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX

        Arguments:
        X -- input dataset placeholder, of shape (input size, number of examples)
        parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                      the shapes are given in initialize_parameters
        Returns:
        Z3 -- the output of the last LINEAR unit
        """

        # Retrieve parameters from dictionary
        W1, b1 = parameters['W1'], parameters['b1']
        W2, b2 = parameters['W2'], parameters['b2']
        W3, b3 = parameters['W3'], parameters['b3']

        # Carry out forward propagation
        Z1 = tf.add(tf.matmul(W1, X), b1)
        A1 = tf.nn.relu(Z1)
        Z2 = tf.add(tf.matmul(W2, A1), b2)
        A2 = tf.nn.relu(Z2)
        Z3 = tf.add(tf.matmul(W3, A2), b3)

        return Z3

    def compute_cost(self, Z3, Y):
        """
        Computes the cost

        Arguments:
        Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (10, number_of_examples)
        Y -- "true" labels vector placeholder, same shape as Z3

        Returns:
        cost - Tensor of the cost function
        """

        # Get logits (predictions) and labels
        logits = tf.transpose(Z3)
        labels = tf.transpose(Y)

        # Compute cost
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))

        return cost

    def model(self, train, test, learning_rate=0.0001, num_epochs=16, minibatch_size=32, print_cost=True,
              graph_filename='costs'):
        """
        Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.

        Arguments:
        train -- training set
        test -- test set
        learning_rate -- learning rate of the optimization
        num_epochs -- number of epochs of the optimization loop
        minibatch_size -- size of a minibatch
        print_cost -- True to print the cost every epoch

        Returns:
        parameters -- parameters learnt by the model. They can then be used to predict.
        """

        # Ensure that model can be rerun without overwriting tf variables
        ops.reset_default_graph()
        # For reproducibility
        tf.set_random_seed(42)
        seed = 42
        # Get input and output shapes
        (n_x, m) = train.images.T.shape
        n_y = train.labels.T.shape[0]

        costs = []

        # Create placeholders of shape (n_x, n_y)
        X, Y = self.create_placeholders(n_x, n_y)
        # Initialize parameters
        parameters = self.initialize_parameters()

        # Forward propagation
        Z3 = self.forward_propagation(X, parameters)
        # Cost function
        cost = self.compute_cost(Z3, Y)
        # Backpropagation (using Adam optimizer)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

        # Initialize variables
        init = tf.global_variables_initializer()

        # Start session to compute Tensorflow graph
        with tf.Session() as sess:

            # Run initialization
            sess.run(init)

            # Training loop
            for epoch in range(num_epochs):

                epoch_cost = 0.
                num_minibatches = int(m / minibatch_size)
                seed = seed + 1

                for i in range(num_minibatches):
                    # Get next batch of training data and labels
                    minibatch_X, minibatch_Y = train.next_batch(minibatch_size)

                    # Execute optimizer and cost function
                    _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X.T, Y: minibatch_Y.T})

                    # Update epoch cost
                    epoch_cost += minibatch_cost / num_minibatches

                # Print the cost every epoch
                if print_cost:
                    print("Cost after epoch {epoch_num}: {cost}".format(epoch_num=epoch, cost=epoch_cost))
                    costs.append(epoch_cost)

            # Plot costs
            plt.figure(figsize=(16, 5))
            plt.plot(np.squeeze(costs), color='#2A688B')
            plt.xlim(0, num_epochs - 1)
            plt.ylabel("cost")
            plt.xlabel("iterations")
            plt.title("learning rate = {rate}".format(rate=learning_rate))
            plt.savefig(graph_filename, dpi=300)
            plt.show()

            # Save parameters
            parameters = sess.run(parameters)
            print("Parameters have been trained!")

            # Calculate correct predictions
            correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

            # Calculate accuracy on test set
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

            print("Train Accuracy:", accuracy.eval({X: train.images.T, Y: train.labels.T}))
            print("Test Accuracy:", accuracy.eval({X: test.images.T, Y: test.labels.T}))

            return parameters


def use_TfNN(fashion=True):
    # Import Fashion MNIST
    if fashion:
        mnist = input_data.read_data_sets('../Datasets/FashionMNIST', one_hot=True)
    else:
        mnist = input_data.read_data_sets('../Datasets/MNIST', one_hot=True)

    # Shapes of training set
    print("Training set (images) shape: {shape}".format(shape=mnist.train.images.shape))
    print("Training set (labels) shape: {shape}".format(shape=mnist.train.labels.shape))

    # Shapes of test set
    print("Test set (images) shape: {shape}".format(shape=mnist.test.images.shape))
    print("Test set (labels) shape: {shape}".format(shape=mnist.test.labels.shape))
    train = mnist.train
    test = mnist.test
    NNclassifieur = TfNN(mnist)
    parameters = NNclassifieur.model(train, test, learning_rate=5e-4)


# TODO tensorflow/keras convolution NN


class KerasResnet50:
    """
    Inspired from https://github.com/tensorflow/models/tree/master/official/resnet.
    """
    def __init__(self):
        self.L2_WEIGHT_DECAY = 1e-4
        self.BATCH_NORM_DECAY = 0.9
        self.BATCH_NORM_EPSILON = 1e-5
        self.model = self.resnet50(10)

    def identity_block(self, input_tensor, kernel_size, filters, stage, block):
        """The identity block is the block that has no conv layer at shortcut.

        # Arguments
            input_tensor: input tensor
            kernel_size: default 3, the kernel size of
                middle conv layer at main path
            filters: list of integers, the filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names

        # Returns
            Output tensor for the block.
        """
        filters1, filters2, filters3 = filters
        if backend.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = layers.Conv2D(filters1, (1, 1), use_bias=False,
                          kernel_initializer='he_normal',
                          kernel_regularizer=regularizers.l2(self.L2_WEIGHT_DECAY),
                          name=conv_name_base + '2a')(input_tensor)
        x = layers.BatchNormalization(axis=bn_axis,
                                      momentum=self.BATCH_NORM_DECAY,
                                      epsilon=self.BATCH_NORM_EPSILON,
                                      name=bn_name_base + '2a')(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(filters2, kernel_size,
                          padding='same', use_bias=False,
                          kernel_initializer='he_normal',
                          kernel_regularizer=regularizers.l2(self.L2_WEIGHT_DECAY),
                          name=conv_name_base + '2b')(x)
        x = layers.BatchNormalization(axis=bn_axis,
                                      momentum=self.BATCH_NORM_DECAY,
                                      epsilon=self.BATCH_NORM_EPSILON,
                                      name=bn_name_base + '2b')(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(filters3, (1, 1), use_bias=False,
                          kernel_initializer='he_normal',
                          kernel_regularizer=regularizers.l2(self.L2_WEIGHT_DECAY),
                          name=conv_name_base + '2c')(x)
        x = layers.BatchNormalization(axis=bn_axis,
                                      momentum=self.BATCH_NORM_DECAY,
                                      epsilon=self.BATCH_NORM_EPSILON,
                                      name=bn_name_base + '2c')(x)

        x = layers.add([x, input_tensor])
        x = layers.Activation('relu')(x)
        return x

    def conv_block(self, input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
        """A block that has a conv layer at shortcut.

        # Arguments
            input_tensor: input tensor
            kernel_size: default 3, the kernel size of
                middle conv layer at main path
            filters: list of integers, the filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names
            strides: Strides for the second conv layer in the block.

        # Returns
            Output tensor for the block.

        Note that from stage 3,
        the second conv layer at main path is with strides=(2, 2)
        And the shortcut should have strides=(2, 2) as well
        """
        filters1, filters2, filters3 = filters
        if backend.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = layers.Conv2D(filters1, (1, 1), use_bias=False,
                          kernel_initializer='he_normal',
                          kernel_regularizer=regularizers.l2(self.L2_WEIGHT_DECAY),
                          name=conv_name_base + '2a')(input_tensor)
        x = layers.BatchNormalization(axis=bn_axis,
                                      momentum=self.BATCH_NORM_DECAY,
                                      epsilon=self.BATCH_NORM_EPSILON,
                                      name=bn_name_base + '2a')(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(filters2, kernel_size, strides=strides, padding='same',
                          use_bias=False, kernel_initializer='he_normal',
                          kernel_regularizer=regularizers.l2(self.L2_WEIGHT_DECAY),
                          name=conv_name_base + '2b')(x)
        x = layers.BatchNormalization(axis=bn_axis,
                                      momentum=self.BATCH_NORM_DECAY,
                                      epsilon=self.BATCH_NORM_EPSILON,
                                      name=bn_name_base + '2b')(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(filters3, (1, 1), use_bias=False,
                          kernel_initializer='he_normal',
                          kernel_regularizer=regularizers.l2(self.L2_WEIGHT_DECAY),
                          name=conv_name_base + '2c')(x)
        x = layers.BatchNormalization(axis=bn_axis,
                                      momentum=self.BATCH_NORM_DECAY,
                                      epsilon=self.BATCH_NORM_EPSILON,
                                      name=bn_name_base + '2c')(x)

        shortcut = layers.Conv2D(filters3, (1, 1), strides=strides, use_bias=False,
                                 kernel_initializer='he_normal',
                                 kernel_regularizer=regularizers.l2(self.L2_WEIGHT_DECAY),
                                 name=conv_name_base + '1')(input_tensor)
        shortcut = layers.BatchNormalization(axis=bn_axis,
                                             momentum=self.BATCH_NORM_DECAY,
                                             epsilon=self.BATCH_NORM_EPSILON,
                                             name=bn_name_base + '1')(shortcut)

        x = layers.add([x, shortcut])
        x = layers.Activation('relu')(x)
        return x

    def resnet50(self, num_classes):
        """Instantiates the ResNet50 architecture.

        Args:
          num_classes: `int` number of classes for image classification.

        Returns:
            A Keras model instance.
        """
        input_shape = (28, 28, 1)
        img_input = layers.Input(shape=input_shape)

        if backend.image_data_format() == 'channels_first':
            x = layers.Lambda(lambda x: backend.permute_dimensions(x, (0, 3, 1, 2)),
                              name='transpose')(img_input)
            bn_axis = 1
        else:  # channels_last
            x = img_input
            bn_axis = 3

        x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(x)
        x = layers.Conv2D(64, (7, 7),
                          strides=(2, 2),
                          padding='valid', use_bias=False,
                          kernel_initializer='he_normal',
                          kernel_regularizer=regularizers.l2(self.L2_WEIGHT_DECAY),
                          name='conv1')(x)
        x = layers.BatchNormalization(axis=bn_axis,
                                      momentum=self.BATCH_NORM_DECAY,
                                      epsilon=self.BATCH_NORM_EPSILON,
                                      name='bn_conv1')(x)
        x = layers.Activation('relu')(x)
        x = layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
        x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

        x = self.conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
        x = self.identity_block(x, 3, [64, 64, 256], stage=2, block='b')
        x = self.identity_block(x, 3, [64, 64, 256], stage=2, block='c')

        x = self.conv_block(x, 3, [128, 128, 512], stage=3, block='a')
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='b')
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='c')
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='d')

        x = self.conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

        x = self.conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
        x = self.identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
        x = self.identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = layers.Dense(
            num_classes, activation='softmax',
            kernel_regularizer=regularizers.l2(self.L2_WEIGHT_DECAY),
            bias_regularizer=regularizers.l2(self.L2_WEIGHT_DECAY),
            name='fc{0}'.format(num_classes))(x)

        # Create model.
        return models.Model(img_input, x, name='resnet50')

    def train(self, num_epochs, mnist):
        train_X, val_X, train_Y, val_Y = mnist

        self.model.compile(loss='categorical_crossentropy',
                      optimizer='Adam',
                      metrics=['categorical_accuracy'])
        history = self.model.fit(train_X, train_Y, epochs=num_epochs, batch_size=128, verbose=1, validation_data=(val_X, val_Y))
        return history


def use_KerasResnet(fashion=True, num_epochs=5):
    if fashion:
        X, X_test, Y, Y_test = mnistProcessing.load_FashionMNIST(one_hot=True)
    else:
        X, X_test, Y, Y_test = mnistProcessing.load_MNIST(one_hot=True)

    model = KerasResnet50()
    history = model.train(num_epochs, (X, X_test, Y, Y_test))


if __name__ == "__main__":
    from DataProcessing import IrisProcessing, mnistProcessing
    # use_NumpyNN(IrisProcessing.load_iris(0.1), epsilon=1e-5, nn_hdim=1024, num_passes=5000)
    # use_NumpyNN(mnistProcessing.load_MNIST(flatten=True, one_hot=True), nn_hdim=256, num_passes=500)
    # use_TfNN()
    use_KerasResnet()
