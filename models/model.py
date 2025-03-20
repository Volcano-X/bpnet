import numpy as np
from utils import CE_loss_acc
from layers import FullyConnectedLayer


class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.loss_fn = None
        self.optimizer = None

    def add(self, layer):
        self.layers.append(layer)

    def set_loss_function(self, loss_fn):
        self.loss_fn = loss_fn

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def forward(self, X):
        Z = X
        for layer in self.layers:
            Z = layer.forward(Z)
        return Z

    def backward(self, d_Z):
        for layer in reversed(self.layers):
            d_Z = layer.backward(d_Z)
        return d_Z

    def train(self, X, Y, batch_size, epochs):
        m = X.shape[0]
        batch_max = m // batch_size
        index = list(range(m))

        for epoch in range(epochs):
            np.random.shuffle(index)

            Z = self.forward(X)
            loss, acc = CE_loss_acc(Z, Y)

            print(
                f"Epoch {epoch}/{epochs:<10}  Loss: {loss:<10.5}  Accuracy: {acc*100:<10.4}"
            )
            for batch in range(batch_max):
                batch_index = index[batch * batch_size : (batch + 1) * batch_size]
                X_batch = X[batch_index, :]
                Y_batch = Y[batch_index, :]

                Z_batch = self.forward(X_batch)
                loss = self.loss_fn.forward(Z_batch, Y_batch)

                d_Z = self.loss_fn.backward()
                self.backward(d_Z)

                for layer in self.layers:
                    if isinstance(layer, FullyConnectedLayer):
                        self.optimizer.step(layer)
