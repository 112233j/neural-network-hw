#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import activations
from optimizer import SGD
from MLP import MLP
from load_data import load_mnist

np.random.seed(2)

epochs = 80
batch_size = 16
learning_rates = [5e-3, 1e-2, 2e-2]
weight_decaies = [0, 1e-2, 2e-2]
dims = [[784] + [i] + [10] for i in range(20,60,10)]

print("training parameters:")
print(f"epochs:{epochs}")
print(f"batch_size:{batch_size}")
print(f"learning_rates:{learning_rates}")
print(f"weight_decaies:{weight_decaies}")
print(f"dims:{dims}")
print("-----------------------------------------------------------")
print("loading data.......")

train_data, val_data, test_data = load_mnist()

print("loading done......")
print("-----------------------------------------------------------")
print("searching best parameters.....")

best_parameters = {"best_accuracy":0,"dim":[784, 20, 10],"learning_rate":0.005,"weight_decay":0}

for dim in dims:
    for lr in learning_rates:
        for weight_decay in weight_decaies:
            print(f"current parameters: learning_rate={lr}   weight_decay={weight_decay}   dims:{dim}")
            print(f"best parameters:{best_parameters}")
            model = MLP(dim)
            optimizer = SGD(model, learning_rate=lr, weight_decay=weight_decay, batch_size=batch_size)
            accuracy = model.fit(train_data, val_data, optimizer, epochs=epochs, batch_size=batch_size, mode=0)
            if accuracy > best_parameters["best_accuracy"]:
                best_parameters["best_accuracy"] = accuracy
                best_parameters["dim"] = dim
                best_parameters["learning_rate"] = lr
                best_parameters["weight_decay"] = weight_decay

print(best_parameters)

# 最佳模型为 model_{50}_{0.02}_{0}

# 模型存储路径为 os.path.join(os.curdir,"models",f"model_{hidden}_{lr}_{weight_decay}.npz")
# 模型加载
model.load(os.path.join(os.curdir,"models","model_50_0.02_0.npz"))

model.predict(test_data)

def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    ax1.plot(history['train_losses'], label='train loss')
    ax1.plot(history['validation_losses'], label='validation loss')

    ax1.set_ylim([-0.05, 0.6])
    ax1.legend()
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Epoch')

    ax2.plot(history['accuracies'], label='validation accuracy')

    ax2.set_ylim([-0.05, 1.05])
    ax2.legend()
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Epoch')

    fig.suptitle('Training History')
    plt.savefig("pictures/Training History.png")
    plt.show()
    plt.close()

def plot_parameters(model):
    for i ,weight in enumerate(model.weights[1:]):
        weight = weight.flatten().tolist()
        plt.hist(weight)
        plt.title(f"layer{i+1} weights")
        plt.xlabel("value")
        plt.ylabel("frequency")
        plt.savefig(f"pictures/layer{i+1}_weights.png")
        plt.show()
        plt.close()
    for i ,bias in enumerate(model.bias[1:]):
        bias = bias.flatten().tolist()
        plt.hist(bias)
        plt.title(f"layer{i+1} bias")
        plt.xlabel("value")
        plt.ylabel("frequency")
        plt.savefig(f"pictures/layer{i+1}_bias.png")
        plt.show()
        plt.close()

# 模型训练历史存储路径为 os.path.join(os.curdir,"logs",f"model_{hidden}_{lr}_{weight_decay}.npz")
history = pd.read_csv(os.path.join(os.curdir,"logs","log_50_0.02_0.csv"))

plot_training_history(history)

plot_parameters(model)

