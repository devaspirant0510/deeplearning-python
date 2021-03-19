import numpy as np


class Layer(object):
    input: object
    output: object

    def __init__(self):
        self.w = None
        self.b = None

    def forward(self, x):
        self.input = x
        self.output = self._forward()
        return self.output

    def _forward(self):
        raise NotImplementedError

    def backward(self, x):
        if self.input is None or self.output is None:
            return None
        return self._backward()

    def _backward(self):
        raise NotImplementedError


class Perceptron(Layer):

    def __init__(self, w, b):
        super().__init__()
        self.w = w
        self.b = b

    def _forward(self):
        self.z = np.dot(self.input, self.w) + self.b
        return self.z

    def _backward(self):
        pass


class Sigmoid(Layer):
    def __init__(self):
        super().__init__()

    def _forward(self):
        self.z = 1 / (1 + np.exp(-self.input))
        return self.z

    def _backward(self):
        self.dL = (1 - self.z) * self.z
        return self.dL
