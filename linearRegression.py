import random

class Model:

    def __init__(self,dataset,outputs):
        self.variables = []
        self.variableCount = len(dataset[0])+1
        self.dataset = dataset
        self.outputs = outputs
        self.count = len(dataset)
        self.lr = 0.001
        for i in range(0,self.variableCount):
            self.variables.append(random.random()*10)

    def predict(self,data):
        output = 0
        for i in range(0,len(data)):
            output += self.variables[i]*data[i]
        output += self.variables[-1]
        return output


    def train(self):

        # here defines the convergence condition
        for m in range(0,80000):

            # iterate and compute cost for each variable
            costs = []
            for k in range(0,self.variableCount):
                costs.append(0.0)
            for i in range(0,self.count):
                data = self.dataset[i]
                value = self.predict(data) - self.outputs[i]
                costs[-1] += value
                for j in range(0,len(data)):
                    costs[j] += value*data[j]
            for a in range(0,len(costs)):
                costs[a] /= self.count
            # optimizing variable values
            for b in range(0,len(self.variables)):
                self.variables[b] = self.variables[b] - self.lr*costs[b]


dataset = [[2.0],[3.0],[2.3],[3.6]]
outputs = [20000.0,30000.0,23000.0,36000.0]

gpaSalary = Model(dataset,outputs)


print(gpaSalary.predict([2.9]))

gpaSalary.train()

print(gpaSalary.predict([2.9]))
print(gpaSalary.predict([3.6]))

# testing with a actual dataset
import numpy as np
import matplotlib.pyplot as plt 

import pandas as pd  
import seaborn as sns 

from sklearn.datasets import load_boston
boston_dataset = load_boston()

boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
print(boston.head())

boston['MEDV'] = boston_dataset.target

X = pd.DataFrame(np.c_[boston['LSTAT'], boston['RM']], columns = ['LSTAT','RM'])
Y = boston['MEDV']

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)

# converting data into python list objects as my implemetation only works
# python lists

X_train = X_train.values.tolist()
X_test = X_test.values.tolist()
Y_train = Y_train.values.tolist()
Y_test = Y_test.values.tolist()

lin_model = Model(X_train, Y_train)
print(">>>>>>>>>>")
print("Before Training")
predictions = []
for data in X_test:
    predictions.append(lin_model.predict(data))
print("Actual Result\tPredicted Results")
for i in range(0,len(predictions)):
    print(Y_test[i],"\t",predictions[i])
print("...................")
print("...................")
print("...................")
lin_model.train()
print("After Training")
predictions = []
for data in X_test:
    predictions.append(lin_model.predict(data))
print("Actual Result\tPredicted Results")
for i in range(0,len(predictions)):
    print(Y_test[i],"\t",predictions[i])


