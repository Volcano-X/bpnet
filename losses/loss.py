import numpy as np
from utils import softmax


class CrossEntropyLoss:
    def __init__(self):
        self.Y = None
        self.Y_pred = None

    def forward(self, Z, Y):
        # Z: final output, Y: one-hot label
        self.Y = Y
        self.Y_pred = softmax(Z)
        temp = self.Y_pred[Y == 1]
        loss = -np.mean(np.log(temp + 1e-8))
        return loss

    def backward(self):
        m = self.Y.shape[0]
        self.d_Z = (self.Y_pred - self.Y) / m
        return self.d_Z
