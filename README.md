# Bpnet

Written by zhengjiacan, Shenzhen University, 2025/03/22.

## Overview

Bpnet is a fully connected neural network that uses stochastic mini-batch gradient descent based back propagation for parameter learning.

This project writes object-oriented class structure for bpnet based on numpy library. It creates a set of classes that each encapsulate different parts of the neural network. The main idea is to structure the code in a modular way, like pytorch, where each component (like layers, loss functions, optimizers, etc.) is an object that interacts with others through well-defined interfaces.

Here’s a basic outline for project structure into object-oriented classes:

+ NeuralNetwork: Main class that manages the layers and the training process.
+ Layer: Base class for all layers in the network (e.g., FullyConnectedLayer, ActivationLayer).
+ Activation Functions: Different activation functions (e.g., ReLU, Sigmoid, Tanh).
+ Loss Function: Contains methods for computing the loss (e.g., CrossEntropyLoss, MSE).
Optimizer: Implements optimization algorithms (e.g., SGD, Adam).
+ Utils: Helper functions like softmax, etc.

This project contains the following file:

+ data: ia folder which contains four csv files: mnist_images_train.csv, mnist_label_train.zip, mnist_images_test.csv, mnist_labels_test.csv. You need to unzip the `mnist_label_traim.zip` first since the file is too large.

## Mathematical Derivation

I mainly untilize Jacobia matrix for [Mathematical Derivation](https://volcano-x.github.io/2024/12/07/bp-network/) of bpnet. One can check the mathematical derivation first at the following url:

    https://volcano-x.github.io/2024/12/07/bp-network/

## How to run

Unzip `mnist_label_train.zip` first to get `mnist_label_train.csv`, then run the following code:

```bash
cd bpnet
python main.py
```
