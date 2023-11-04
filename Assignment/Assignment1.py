import numpy as np
import math

class Plus():

    def __init__(self):
        pass

    def forward(self, x, y):
        return x + y

    def backward(self, goutput):
        return goutput, goutput

class Sigmoid():

    def __init__(self):
        pass

    def forward(self, x):
        sigx = 1 / (1 + np.exp(-x))
        self.sigx = sigx
        return sigx

    def backward(self, goutput):
        sigx = self.sigx
        return goutput * sigx * (1 - sigx)

class Softmax():

    def __init__(self):
        pass

    def forward(self, x):
        sofx = [math.exp(i) / sum([math.exp(j) for j in x]) for i in x]
        self.sofx = sofx
        return sofx

    def backward(self, soutput):
        sofx = self.sofx
        gradients = []
        for i in range(len(sofx)):
            gradient = sofx[i] * (1 - sofx[i])
            if i == soutput:
                gradient -= 1.0
            gradients.append(gradient)
        return gradients

class RowSum():

    def __init__(self):
        pass

    def forward(self, x):
        sumd = x.sum(axis=1)
        self.x = x.shape[1]
        return sumd

    def backward(self, gy):
        n, m= gy.shape[0], self.x
        return gy[:,None].expand(n, m)

class Expand():

    def __init__(self):
        pass

    def forward(self, x, size):
        return np.full(x, size=size)

    def backward(self, gy):
        return gy.sum(), None

# Input layer
inputs = [1.,-1.]

# First layer
W = [[1., 1., 1.], [-1., -1., -1.]]
b = [0., 0., 0.]

# Second layer
V = [[1., 1.], [-1., -1.], [-1., -1.]]
c = [0., 0.]

# Sigmoid activation function
sigmoid = Sigmoid()

# Softmax activation function
softmax = Softmax()

# Forward pass
k = np.dot(inputs, W) + b
h = sigmoid.forward(k)
s = np.dot(h, V) + c
o = softmax.forward(s)
loss = -np.log(o[1])

# Backward pass
