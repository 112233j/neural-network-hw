#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import activations
import random
import os
import time

class MLP:
    def __init__(self, dim):
        self.dim = dim
        self.num_layers = len(dim)
        self.weights = [np.array([0])] + [ np.random.randn(dim[i], dim[i-1])/np.sqrt(dim[i-1]) for i in range(1,len(dim)) ]
        self.bias = [np.array([0])] + [ np.random.randn(dim[i],1) for i in range(1,len(dim)) ]
        
        self.linears = [np.zeros(layer.shape) for layer in self.bias]
        self.activations = [np.zeros(layer.shape) for layer in self.bias]
    
    def forward(self, input):
        # input = (n,1)
        self.activations[0] = input
        for i in range(1, self.num_layers):
            self.linears[i] = self.weights[i].dot(self.activations[i-1]) + self.bias[i]
            if i == self.num_layers - 1:
                self.activations[i] = activations.softmax(self.linears[i])
            else:
                self.activations[i] = activations.ReLU(self.linears[i])
        
        return self.activations[-1]
        
    def backward(self, loss_gradient):
        # loss_gradient 为交叉熵损失函数和softmax的导数
        self.delta_w = [np.zeros(w.shape) for w in self.weights]
        self.delta_b = [np.zeros(b.shape) for b in self.bias]
        # 梯度反向传播
        self.delta_b[-1] = loss_gradient
        self.delta_w[-1] = loss_gradient.dot(self.activations[-2].transpose())
        for i in range(self.num_layers-2, 0, -1):
            self.delta_b[i] = np.multiply(self.weights[i+1].transpose().dot(self.delta_b[i+1]), 
                                          activations.ReLU_gradient(self.linears[i]))
            self.delta_w[i] = self.delta_b[i].dot(self.activations[i-1].transpose())
            
        return self.delta_w, self.delta_b
    def fit(self, train_data, validation_data, optimizer, epochs, batch_size, mode=0):

        train_losses = []
        validation_losses = []
        best_accuracy = 0
        accuracies = []
        
        self.optimizer = optimizer
        start_time = time.time()
        for epoch in range(1, epochs+1):
            # train
            random.shuffle(train_data)
            batches = [train_data[k:k+optimizer.batch_size] for k in range(0, len(train_data), optimizer.batch_size)]
            train_loss = 0
            for batch in batches:
                self.optimizer.zero_gradient()
                for x, y in batch:
                    pred = self.forward(x)
                    # loss为交叉熵损失函数
                    train_loss  += np.where(y==1, -np.log(pred), 0).sum()
                    # cross entropy loss + softmax 的导数
                    loss_gradient = pred - y
                    delta_w, delta_b = self.backward(loss_gradient)
                    self.optimizer.update(delta_w, delta_b)
                self.optimizer.step()
            train_loss /= len(train_data)
            train_losses.append(train_loss)

            # validate
            validation_loss = 0
            accuracy = 0
            for x, y in validation_data:
                pred = self.forward(x)
                validation_loss += np.where(y==1, -np.log(pred), 0).sum()
                accuracy += np.where(y==1, pred, 0).sum()
            validation_loss /= len(validation_data)
            validation_losses.append(validation_loss)
            accuracy /= len(validation_data) 
            accuracies.append(accuracy)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self.save(os.path.join(os.curdir,"models",f"model_{self.dim[1]}_{self.optimizer.lr}_{self.optimizer.weight_decay}.npz"))
            if epoch % 20 ==0:
                if mode == 1:
                    print("current_epoch:",epoch)
                    print("current_train_loss:",train_loss)
                    print("current_validation_loss:",validation_loss)
                    print("current_accuracy:",accuracy,"   best_accuracy:",best_accuracy,"   Time:",time.time()-start_time)
                if mode == 0:
                    print("current_epoch:",epoch)
                    print("current_accuracy:",accuracy,"   best_accuracy:",best_accuracy,"   Time:",time.time()-start_time)
        print("----------------------------------------------------------------------------------------------")

        training_log = {"train_losses":train_losses,
                 "validation_losses":validation_losses ,
                 "accuracies":accuracies,
                 "best_accuracy":best_accuracy}
        pd.DataFrame(training_log).to_csv(f'logs/log_{self.dim[1]}_{self.optimizer.lr}_{self.optimizer.weight_decay}.csv',)   
        return best_accuracy
    
    def predict(self,test_data):
        test_loss = 0
        accuracy = 0
        for x, y in test_data:
            pred = self.forward(x)
            test_loss += np.where(y==1, -np.log(pred), 0).sum()
            accuracy += np.where(y==1, pred, 0).sum()
        test_loss /= len(test_data)
        accuracy /= len(test_data) 
        print("predicted loss:",test_loss)
        print("predicted accuracy:",accuracy)
    
    
    def save(self,filename):
        np.savez(file=filename,
           weights = self.weights,
           bias = self.bias,
           linears = self.linears,
           activations = self.activations)
    
    def load(self, filename):
        parameters = np.load(filename, allow_pickle=True)
        
        self.weights = list(parameters["weights"])
        self.bias = list(parameters["bias"])
        self.linear = list(parameters["linears"])
        self.activations = list(parameters["activations"])





