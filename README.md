# neural-network-hw
实验选择超参数如下：  
epochs:80  
batch_size:16  
learning_rates:[0.005, 0.01, 0.02]  
weight_decaies:[0, 0.01, 0.02]  
dims:[[784, 20, 10], [784, 30, 10], [784, 40, 10], [784, 50, 10]]

最终准确率最高的模型为：  
best_accuracy: 0.9666661314755447  
dim: [784, 50, 10]  
learning_rate: 0.02  
weight_decay: 0  

模型保存在"..\models\model_50_0.02_0.npz"

加载模型
model.load(os.path.join(os.curdir,"models","model_50_0.02_0.npz"))
预测
model.predict(test_data)对测试机预测
![image](https://user-images.githubusercontent.com/79825105/230629038-92fc604d-b63f-425d-93b3-d57104f8117e.png)
![image](https://user-images.githubusercontent.com/79825105/230629053-a3659bd0-ba8a-4ee9-acbb-12f6a02df803.png)
![image](https://user-images.githubusercontent.com/79825105/230629069-eff4a377-1b28-4805-96f0-b25c9a17f346.png)
![image](https://user-images.githubusercontent.com/79825105/230629090-ea350170-3204-43c2-8e57-8a0016615adf.png)
![image](https://user-images.githubusercontent.com/79825105/230629111-4b37f577-c7b1-4f31-9dac-7135419e7a9b.png)
