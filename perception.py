import math
import random

class Perceptron:

    def __init__(self,inputs):
        self.weights = []
        self.bais = random.random()
        for i in range(0,inputs):
            self.weights.append(random.random())
            
