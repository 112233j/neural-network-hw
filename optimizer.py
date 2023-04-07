#!/usr/bin/env python
# coding: utf-8
import numpy as np

class SGD:
    def __init__(self, model, learning_rate, weight_decay, batch_size):
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.model = model
        #储存batch内的梯度变化量
        self.delta_w = [np.zeros(w.shape) for w in self.model.weights]
        self.delta_b = [np.zeros(b.shape) for b in self.model.bias]
        
    def zero_gradient(self):
        self.delta_w = [np.zeros(w.shape) for w in self.model.weights]
        self.delta_b = [np.zeros(b.shape) for b in self.model.bias]
    
    def update(self, delta_w, delta_b):
        self.delta_w = [w + dw for w,dw in zip(self.delta_w, delta_w)]
        self.delta_b = [b + db for b,db in zip(self.delta_b, delta_b)]
        
    def step(self):
        self.model.weights = [(1-self.weight_decay*self.lr)*w - (self.lr/self.batch_size) * dw for w,dw in zip(self.model.weights, self.delta_w)]
        self.model.bias =   [(1-self.weight_decay*self.lr)*b - (self.lr/self.batch_size) * db for b,db in zip(self.model.bias, self.delta_b)]

