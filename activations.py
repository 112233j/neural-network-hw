#!/usr/bin/env python
# coding: utf-8

import numpy as np


def ReLU(input):
    return np.maximum(0,input)

def ReLU_gradient(input):
    return input > 0

def softmax(input):
    return np.exp(input)/np.sum(np.exp(input))

