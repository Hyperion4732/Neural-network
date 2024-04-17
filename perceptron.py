import numpy as np


class Perceptron:
    def __init__(self, layer_sizes, features, targets, learning_rate, output=None, layers=None, weights=None):
        self.layer_sizes = layer_sizes
        self.features = features
        self.targets = targets
        self.learning_rate = learning_rate
        self.output = output
        self.layers = layers
        self.weights = weights

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)

    def MSE(self):
        MSE = 0
        for i in range(len(self.targets)):
            MSE += ((self.targets[i] - self.output[i]) ** 2) / len(self.targets)
        return MSE

    def create_weights_matrix(self):
        self.weights = [np.random.rand(self.layer_sizes[i], self.layer_sizes[i + 1]) for i in range(len(self.layer_sizes) - 1)]

    def forward_pass(self):
        layers = [self.features]
        for i in range(len(self.weights)):
            layers.append(self.sigmoid(np.dot(layers[i], self.weights[i])))
        self.layers = layers

    def back_propagation(self):
        output = self.layers[-1]
        error = self.targets - output
        deltas = [error * self.sigmoid_derivative(output)]

        for layer in range(len(self.layer_sizes) - 2, 0, -1):
            error = deltas[0].dot(self.weights[layer].T)
            delta = error * self.sigmoid_derivative(self.layers[layer])
            deltas.insert(0, delta)

        for i in range(len(self.weights)):
            self.weights[i] += self.layers[i].T.dot(deltas[i]) * self.learning_rate

        self.output = output

    def train(self, epochs):
        self.create_weights_matrix()
        for i in range(epochs):
            for j in range(len(self.features)):
                self.forward_pass()
                self.back_propagation()
                print(f'Эпоха {i}: MSE = {self.MSE()}')

    def show_result(self):
        print(self.output)
