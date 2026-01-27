import math

class Neuron:
    def __init__(self, weight1, weight2, bias, regression=False):
        self.weights_1 = weight1
        self.weights_2 = weight2
        self.bias = bias
        self.regression = regression

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))
    
    def feedforward(self, x1, x2):
        z = (x1 * self.weights_1) + (x2 * self.weights_2) + self.bias
        if self.regression:
            return z
        else:
            return self.sigmoid(z)

class OurNeuralNetwork:
    def __init__(self, neuron_h1, neuron_h2, neuron_o1):
        self.h1 = neuron_h1
        self.h2 = neuron_h2
        self.o1 = neuron_o1

    def feedforward(self, x1, x2):

        # hado homa Hidden layer
        h1_out = self.h1.feedforward(x1, x2)
        h2_out = self.h2.feedforward(x1, x2)

        # o hada howa Output layer
        y = self.o1.feedforward(h1_out, h2_out)
        return y
