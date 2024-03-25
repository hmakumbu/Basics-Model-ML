# -*- coding: utf-8 -*-
"""Class_Neural_NetWork.ipynb
"""

import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self):
        pass

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))


    def d_sigmoid(z):
        return NeuralNetwork.sigmoid(z) * (1 - NeuralNetwork.sigmoid(z))


    def loss(y_pred, Y):
        return -np.sum(Y * np.log(y_pred) + (1 - Y) * np.log(1 - y_pred)) / len(Y)

    # Initialize parameters
    def init_params(h0, h1, h2):
        W1 = np.random.randn(h1, h0)
        b1 = np.random.randn(h1, 1)
        W2 = np.random.randn(h2, h1)
        b2 = np.random.randn(h2, 1)
        return W1, W2, b1, b2

    # Forward pass
    def forward_pass(X, W1, W2, b1, b2):
        Z1 = W1.dot(X) + b1
        A1 = NeuralNetwork.sigmoid(Z1)
        Z2 = W2.dot(A1) + b2
        A2 = NeuralNetwork.sigmoid(Z2)
        return A2, Z2, A1, Z1

    # Backward pass
    def backward_pass(X, Y, A2, Z2, A1, Z1, W1, W2, b1, b2):
        m = len(Y)
        dZ2 = A2 - Y
        dW2 = (1 / m) * np.dot(dZ2, A1.T)
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
        dZ1 = np.dot(W2.T, dZ2) * NeuralNetwork.d_sigmoid(Z1)
        dW1 = (1 / m) * np.dot(dZ1, X.T)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
        return dW1, dW2, db1, db2

    # Accuracy
    def accuracy(y_pred, y):
        return np.sum(y_pred == y) / len(y)

    # predict
    def predict(X, W1, W2, b1, b2):
        A2, _, _, _ = NeuralNetwork.forward_pass(X, W1, W2, b1, b2)
        predictions = (A2 >= 0.5).astype(int)
        return predictions

    # Update parameters
    def update(W1, W2, b1, b2, dW1, dW2, db1, db2, alpha):
        W1 -= alpha * dW1
        W2 -= alpha * dW2
        b1 -= alpha * db1
        b2 -= alpha * db2
        return W1, W2, b1, b2

    # Plot decision boundary
    def plot_decision_boundary(W1, W2, b1, b2):
        x = np.linspace(-0.5, 2.5, 100)
        y = np.linspace(-0.5, 2.5, 100)
        xv, yv = np.meshgrid(x, y)
        X_ = np.stack([xv, yv], axis=0)
        X_ = X_.reshape(2, -1)
        A2, _, _, _ = NeuralNetwork.forward_pass(X_, W1, W2, b1, b2)
        plt.figure()
        plt.scatter(X_[0, :], X_[1, :], c=A2)
        plt.show()

    def train(self, X_train, Y_train, X_test, Y_test, alpha=0.01, h0=2, h1=10, h2=1, n_epochs=10000):
        train_loss = []
        test_loss = []
        W1, W2, b1, b2 = NeuralNetwork.init_params(h0, h1, h2)
        for i in range(n_epochs):
            # Forward pass
            A2, Z2, A1, Z1 = NeuralNetwork.forward_pass(X_train, W1, W2, b1, b2)
            # Backward pass
            dW1, dW2, db1, db2 = NeuralNetwork.backward_pass(X_train, Y_train, A2, Z2, A1, Z1, W1, W2, b1, b2)
            # Update parameters
            W1, W2, b1, b2 = NeuralNetwork.update(W1, W2, b1, b2, dW1, dW2, db1, db2, alpha)
            # Save the train loss
            train_loss.append(NeuralNetwork.loss(A2, Y_train))
            # Compute test loss
            A2, Z2, A1, Z1 = NeuralNetwork.forward_pass(X_test, W1, W2, b1, b2)
            test_loss.append(NeuralNetwork.loss(A2, Y_test))
            # Plot boundary
            if i % 1000 == 0:
                NeuralNetwork.plot_decision_boundary(W1, W2, b1, b2)

        # Plot train and test losses
        plt.plot(train_loss)
        plt.plot(test_loss)
        plt.show()

        # Calculate accuracy
        train_pred = NeuralNetwork.predict(X_train, W1, W2, b1, b2)
        train_accuracy = NeuralNetwork.accuracy(train_pred, Y_train)
        print("Train accuracy:", train_accuracy)

        test_pred = NeuralNetwork.predict(X_test, W1, W2, b1, b2)
        test_accuracy = NeuralNetwork.accuracy(test_pred, Y_test)
        print("Test accuracy:", test_accuracy)


if __name__ == "__main__":

    # generate data
    var = 0.2
    n = 800
    class_0_a = var * np.random.randn(n//4,2)
    class_0_b =var * np.random.randn(n//4,2) + (2,2)

    class_1_a = var* np.random.randn(n//4,2) + (0,2)
    class_1_b = var * np.random.randn(n//4,2) +  (2,0)

    X = np.concatenate([class_0_a, class_0_b,class_1_a,class_1_b], axis =0)
    Y = np.concatenate([np.zeros((n//2,1)), np.ones((n//2,1))])
    X.shape, Y.shape

    # shuffle the data
    rand_perm = np.random.permutation(n)

    X = X[rand_perm, :]
    Y = Y[rand_perm, :]

    X = X.T
    Y = Y.T
    X.shape, Y.shape

    # train test split
    ratio = 0.8
    X_train = X [:, :int (n*ratio)]
    Y_train = Y [:, :int (n*ratio)]

    X_test = X [:, int (n*ratio):]
    Y_test = Y [:, int (n*ratio):]

    plt.scatter(X_train[0,:], X_train[1,:], c=Y_train[0,:])
    plt.show()

    nn = NeuralNetwork()
    nn.train(X_train, Y_train, X_test, Y_test)
