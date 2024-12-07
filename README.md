# BPNet

>  Written by Zheng Jiacan, Shenzhen University, 2020/11/1.

## Overview

BPNet is a full-connected multilayer network which uses stochastic gradient descent based back propagation algorithm for optimization. It can be used for classification or regression tasks. This project contains the following file:

+ code / BPNet.py :  a python code implementation based on numpy library. Coding is also according to the derivation of BPNet.pdf.
+ code / test.py :  a python code for testing BPNet.py on minist data set.
+ data: a folder which contains four csv file: minist_images_train.csv, minist_label_train.csv, minist_images_test.csv, minist_labels_test.csv.
+ trainParameters:  a folders which contains a pre-trained parameters.

## Mathematical Derivation

I mainly untilize the Jacobia matrix for [Mathematical Derivation](https://volcano-x.github.io/2024/12/07/bp-network/) of BPNet:

    https://volcano-x.github.io/2024/12/07/bp-network/

## how to run

    ```bash
    cd code
    python3 code/test.py
    ```
