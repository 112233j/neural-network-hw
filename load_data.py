#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os
import gzip
import pickle

def vectorized_result(y):
    e = np.zeros((10, 1))
    e[y] = 1.0
    return e

def load_mnist():
    data_file = gzip.open(os.path.join(os.curdir, "datasets", "mnist.pkl.gz"), "rb")
    train_data, val_data, test_data = pickle.load(data_file, encoding="latin1")
    data_file.close()

    train_inputs = [np.reshape(x, (784, 1)) for x in train_data[0]]
    train_results = [vectorized_result(y) for y in train_data[1]]
    train_data = list(zip(train_inputs, train_results))

    val_inputs = [np.reshape(x, (784, 1)) for x in val_data[0]]
    val_results = [vectorized_result(y) for y in val_data[1]]
    val_data = list(zip(val_inputs, val_results))

    test_inputs = [np.reshape(x, (784, 1)) for x in test_data[0]]
    test_results = [vectorized_result(y) for y in test_data[1]]
    test_data = list(zip(test_inputs, test_results))
    
    return train_data, val_data, test_data

