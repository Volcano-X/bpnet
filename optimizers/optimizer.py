import numpy as np


class SGD:
    def __init__(self, learning_rate=1e-2):
        self.learning_rate = learning_rate
    
    def step(self, layer):
        layer.W -= self.learning_rate * layer.d_W.T
        layer.b -= self.learning_rate * layer.d_b.T
    