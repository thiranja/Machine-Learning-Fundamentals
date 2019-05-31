#import numpy as np
import random
import math

class Nueron:
    def __init__(self):
        self.weights = []
        self.bais = (random.random()-0.5)*10
        self.lastActivation = 0.0

    def sigmoidCalc(self,weightedsum):
        try:
            sigmoidValue = 1/(1 + math.exp(-weightedsum))
        except:
            #print ("Math Value overflow error occured")
            sigmoidValue = 0.0000001
        return sigmoidValue

    def predict(self,inputs):
        weigtedSum = 0.0
        for i in range(0,len(inputs)):
            weigtedSum += self.weights[i]*inputs[i]
        sigValue = self.sigmoidCalc(weigtedSum + self.bais)
        self.lastActivation = sigValue
        return sigValue
    
    def fit(self,cost,weightedSums,lr):
        costs = []
        for i in range(0,len(self.weights)):
            costs.append(cost - cost*weightedSums[i])
            self.weights[i] += costs[i]*lr
        return costs

class Layer:
    def __init__(self,units, inputLength):
        self.nuerons = []
        self.layerId = 0
        for i in range(0,units):
            neuron = Nueron()
            for j in range(0, inputLength):
                val = (random.random() - 0.5)*5
                neuron.weights.append(val)
            self.nuerons.append(neuron)

    def predict(self,inputs):
        output = []
        for neuron in self.nuerons:
            neoronOutput = neuron.predict(inputs)
            output.append(neoronOutput)
        return output

    def fit(self,cost,weightedSums,lr):
        costs = []
        for i in range(0,len(self.nuerons)):
            costs.append(self.nuerons[i].fit(cost[i],weightedSums[i],lr))
        actualCosts = []
        for i in range(0,len(costs[0])):
            temp = 0
            for j in range(0,len(costs)):
                temp += costs[j][i]
            actualCosts.append(temp)
        return actualCosts

class Model:
    def __init__(self,layers,type = 'c'):
        self.lr = 0.1
        self.layers = layers
        for i in range(0,len(self.layers)):
            self.layers[i].layerId = i
        self.layersOutputs = []

    def addSequantial(self,layer):
        self.layers.append(layer)
    
    def predict(self,inputs):
        layersOut = []
        layersOut.append(inputs)
        for layer in self.layers:
            output = layer.predict(layersOut.pop())
            layersOut.append(output)
        layerOutput = layersOut.pop()
        
        #return layerOutput.index(max(layerOutput))
        return layerOutput

    def fit(self,features,labels):
        for i in range(0,len(features)):
            output = self.predict(features[i])
            cost = []
            for k in range(0, len(output)):
                if (labels[i] == k):
                    cost.append(1-output[k])
                else:
                    cost.append(0-output[k])
            self.backpropergate(cost)
            # for j in range(len(self.layers)-1,-1,-1):
            #     # calculate cost
            #     cost = self.layers[j].fit(cost,weightedSums[j],self.lr)
            #     # back propergation
                

    def backpropergate(self,costPara):
        cost = costPara
        for i in range(len(self.layers)-1,-1,-1):
            costGrid = []
            layer = self.layers[i]
            for j in range(0,len(layer.nuerons)):
                costColumn = []
                nuerone = layer.nuerons[j]
                costin = cost.pop(0)
                for k in range(0,len(nuerone.weights)):
                    nudje = nuerone.weights[k]*costin*self.lr
                    nuerone.weights[k]
                    if (nuerone.weights[k] > 0):
                        if (costin > 0):
                            nuerone.weights[k] += nudje
                            if (i != 0):
                                costColumn.append(1-self.layers[i-1].nuerons[k].lastActivation)
                        else:
                            nuerone.weights[k] += nudje
                            if (i != 0):
                                costColumn.append(0-self.layers[i-1].nuerons[k].lastActivation)
                    else:
                        if (costin > 0):
                            nuerone.weights[k] -= nudje
                            if (i != 0):
                                costColumn.append(0-self.layers[i-1].nuerons[k].lastActivation)
                        else:
                            nuerone.weights[k] -= nudje
                            if (i != 0):
                                costColumn.append(1-self.layers[i-1].nuerons[k].lastActivation)
                nuerone.bais += nuerone.bais*costin*self.lr
                #bais edit
                costGrid.append(costColumn)
            # making the cost vector for next layer
            for j in range(0,len(costGrid[0])):
                temp = 0.0
                for k in range(0,len(costGrid)):
                    temp += costGrid[k][j]
                cost.append(temp/len(costGrid))
                
                
                    # b
    
    

                


print ("Hello ML")

l1 = Layer(3,2)
l2 = Layer(4,3)
l3 = Layer(3,4)
cmodel = Model([l1,l2,l3])
print("Gussing [2,5] before training")
output = cmodel.predict([2,5])
print(output)
print("Gussing [6,7] before training")
output = cmodel.predict([6,7])
print(output)
print("Gussing [1,3] before training")
output = cmodel.predict([1,3])
print(output)
print("Gussing [9,1] before training")
output = cmodel.predict([9,1])
print(output)
print()
features = [[2,4],[2,7],[3,9],[6,4]]
labels = [1,1,0,2]
for i in range(0,10):
    cmodel.fit(features,labels)
print("Gussing [2,5] after training")
output = cmodel.predict([2,5])
print(output)

# trying with irs

from sklearn import datasets

iris = datasets.load_iris()
digits = datasets.load_digits()

iris_data = iris.data #variable for array of the data
iris_target = iris.target #variable for array of the labels

import numpy as np
iris_test_ids = np.random.permutation(len(iris_data))

iris_train_one = iris_data[iris_test_ids[:-15]]
iris_test_one = iris_data[iris_test_ids[-15:]]
iris_train_two = iris_target[iris_test_ids[:-15]]
iris_test_two = iris_target[iris_test_ids[-15:]]

la1 = Layer(3,4)
la2 = Layer(4,3)
la3 = Layer(3,4)

clmodel = Model([la1,la2,la3])


predictions = []

for i in range(0,10000):
    clmodel.fit(iris_train_one,iris_train_two)
    if (i % 100 == 0):
        # do predict and print
        for data in iris_test_one:
            print (data)
            predictions.append(clmodel.predict(data))

        print(predictions)
        print(iris_test_two)
        

# for data in iris_test_one:
#     print (data)
#     predictions.append(clmodel.predict(data))

# print(predictions)
# print(iris_test_two)

# for i in range(0,len(iris_test_one)):
#     print("Data is")
#     print(iris_test_one[i])
#     print("Prediction is")
#     print(clmodel.predict(iris_test_one[i]))
#     print("Real Label")
#     print(iris_test_two[i])

