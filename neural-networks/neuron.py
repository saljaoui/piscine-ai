import math

class Neuron:
    def __init__(self, weight1, weight2, bias):
        self.weights_1 = weight1
        self.weights_2 = weight2
        self.bias = bias

    def feedforward(self, x1, x2):
        z = (x1 * self.weights_1) + (x2 * self.weights_2) + self.bias
        return 1 / (1 + math.exp(-z))
