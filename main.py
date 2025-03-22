import pandas as pd
import numpy as np
from layers import FullyConnectedLayer, ReLU
from losses import CrossEntropyLoss
from utils import CE_loss_acc
from models import NeuralNetwork
from optimizers import SGD

if __name__ == "__main__":

    # 将dataframe类型转化为矩阵类型并转置，方便操作
    # 60000*784  S = {0,1,...,254,255} 一个字节表示一个像素，共 60000 个样本，每个样本 784 = 28*28 维度
    # 60000*1    S = {0,1,...,8,9}
    X_train = np.array(pd.read_csv("data/mnist_images_train.csv", header=None))
    labels_train = np.array(pd.read_csv("data/mnist_labels_train.csv", header=None))
    X_test = np.array(pd.read_csv("data/mnist_images_test.csv", header=None))
    labels_test = np.array(pd.read_csv("data/mnist_labels_test.csv", header=None))

    # 转化成 one-hot 标签矩阵 Y
    m_train = labels_train.shape[0]
    m_test = labels_test.shape[0]

    Y_train = np.zeros((m_train, 10))
    Y_train[np.arange(m_train), labels_train[:, 0]] = 1

    Y_test = np.zeros((m_test, 10))
    Y_test[np.arange(m_test), labels_test[:, 0]] = 1

    # define network structure
    model = NeuralNetwork()
    model.add(FullyConnectedLayer(784, 512))
    model.add(ReLU())
    model.add(FullyConnectedLayer(512, 10))

    model.set_loss_function(CrossEntropyLoss())
    model.set_optimizer(SGD(learning_rate=1e-3))

    model.train(X_train, Y_train, batch_size=512, epochs=100)

    Z_test = model.forward(X_test)
    loss, acc = CE_loss_acc(Z_test, Y_test)
    print(f"Test: Loss: {loss:<10.5}  Accuracy: {acc*100:<10.4}")
