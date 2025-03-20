import numpy as np


class Layer:
    def __init__(self):
        # input: Y, output: Z
        self.Y = None
        self.Z = None
        self.W = None
        self.b = None
        self.d_W = None
        self.d_b = None
        self.d_Y = None

    def forward(self, Y):
        raise NotImplementedError

    def backward(self, d_Z):
        raise NotImplementedError


class FullyConnectedLayer(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.W = np.random.normal(
            0,
            np.sqrt(2 / (input_size + output_size)),
            (input_size, output_size),
        )
        self.b = np.zeros((1, output_size))

    def forward(self, Y):
        self.Y = Y
        self.Z = np.dot(Y, self.W) + self.b
        return self.Z

    def backward(self, d_Z):
        m = self.Y.shape[0]
        self.d_W = np.dot(d_Z.T, self.Y)
        self.d_b = np.dot(d_Z.T, np.ones((m, 1)))
        self.d_Y = np.dot(d_Z, self.W.T)
        return self.d_Y


class ReLU:
    def __init__(self):
        # input: Z, output: Y
        self.Z = None
        self.Y = None
        self.d_Z = None

    def forward(self, Z):
        self.Z = Z
        Z[Z < 0] = 0
        self.Y = Z
        return self.Y

    def backward(self, d_Y):
        d_ReLU = self.Y
        d_ReLU[d_ReLU > 0] = 1
        self.d_Z = d_Y * d_ReLU
        return self.d_Z
