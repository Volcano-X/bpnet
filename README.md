# BPNet

> Written by Zheng Jiacan, Shenzhen University, 2020/11/1.

## Overview

BPNet is a full-connected multilayer network which uses stochastic gradient descent based back propagation algorithm for optimization. It can be used for classification or regression tasks. This project contains the following file:

+ code / BPNet.py :  a python code implementation based on numpy library. Coding is also according to the derivation of BPNet.pdf.
+ code / test.py :  a python code for testing BPNet.py on minist data set.
+ data: a folder which contains four csv file: mnist_images_train.csv, mnist_label_train.zip, mnist_images_test.csv, mnist_labels_test.csv.

## Mathematical Derivation

I mainly untilize the Jacobia matrix for [Mathematical Derivation](https://volcano-x.github.io/2024/12/07/bp-network/) of BPNet:

    https://volcano-x.github.io/2024/12/07/bp-network/

## how to run

unzip `mnist_label_train.zip` first to get `mnist_label_train.csv`. then run the following code:

```bash
cd code
python3 code/test.py
```
