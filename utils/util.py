import numpy as np


def softmax(Z):
    # 计算 softmax 的函数值，矢量化计算, Z可以为一个矩阵
    # 因为 max 操作默认转化为行向量，因此需要 keepdims 来保持其为列向量
    Z = Z - Z.max(axis=1, keepdims=True)  # 防止溢出处理
    Z = np.exp(Z)
    Y = Z / Z.sum(axis=1, keepdims=True)
    return Y


def CE_loss_acc(Z, Y):
    Y_pred = softmax(Z)
    temp = Y_pred[Y == 1]
    loss = -np.mean(np.log(temp + 1e-8))

    m = Y.shape[0]
    label_pred = Y_pred.argmax(axis=1)
    result_bool = Y[np.arange(m), label_pred]
    acc = sum(result_bool) / m
    return loss, acc
