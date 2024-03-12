import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self, lr=0.01, n_epochs=1000, batch_size=None):
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.theta = None
        self.loss_history = []

    def add_ones(self, X):
        return np.hstack((np.ones((X.shape[0], 1)), X))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def binary_cross_entropy_loss(self, y_pred, y_true):
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def predict_proba(self, X):
        X = self.add_ones(X)
        return self.sigmoid(np.dot(X, self.theta))

    def predict(self, X, threshold=0.5):
        return self.predict_proba(X) >= threshold

    def accuracy(self, y_true, y_pred):
        return np.mean(y_true == y_pred) * 100

    def fit_batch_gradient_descent(self, X, y):
        X = self.add_ones(X)
        self.theta = np.zeros(X.shape[1])
        self.loss_history = []

        for epoch in range(self.n_epochs):
            y_pred = self.sigmoid(np.dot(X, self.theta))
            grad = np.dot(X.T, (y_pred - y)) / len(y)
            self.theta -= self.lr * grad
            loss = self.binary_cross_entropy_loss(y_pred, y)
            self.loss_history.append(loss)

    def fit_stochastic_gradient_descent(self, X, y):
        X = self.add_ones(X)
        self.theta = np.zeros(X.shape[1])
        self.loss_history = []

        for epoch in range(self.n_epochs):
            for i in range(len(y)):
                xi = X[i].reshape(1, -1)
                yi = y[i]
                y_pred = self.sigmoid(np.dot(xi, self.theta))
                grad = np.dot(xi.T, (y_pred - yi))
                self.theta -= self.lr * grad
            loss = self.binary_cross_entropy_loss(self.sigmoid(np.dot(X, self.theta)), y)
            self.loss_history.append(loss)

    def fit_mini_batch_gradient_descent(self, X, y):
        X = self.add_ones(X)
        self.theta = np.zeros(X.shape[1])
        self.loss_history = []
        batch_size = self.batch_size if self.batch_size is not None else len(y)

        for epoch in range(self.n_epochs):
            for i in range(0, len(y), batch_size):
                xi = X[i:i+batch_size]
                yi = y[i:i+batch_size]
                y_pred = self.sigmoid(np.dot(xi, self.theta))
                grad = np.dot(xi.T, (y_pred - yi)) / len(yi)
                self.theta -= self.lr * grad
            loss = self.binary_cross_entropy_loss(self.sigmoid(np.dot(X, self.theta)), y)
            self.loss_history.append(loss)

    def plot_loss(self):
        plt.plot(range(1, self.n_epochs + 1), self.loss_history)
        plt.xlabel('Epochs')
        plt.ylabel('Binary Cross Entropy Loss')
        plt.title('Training Loss')
        plt.show()