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
    self.sigx = None

  def forward(self, x):
    sigx = 1 / (1 + np.exp(-x))
    self.sigx = sigx
    return sigx

  def backward(self, goutput):
    sigx = self.sigx
    return goutput * sigx * (1 - sigx)

class Softmax():

    def __init__(self):
        self.sofx = None

    def forward(self, x):
        sofx = [math.exp(i) / sum([math.exp(j) for j in x]) for i in x]
        self.sofx = sofx
        return sofx

    def backward(self, soutput):
        gradients = []
        sofx = self.sofx
        for i in range(len(sofx)):
          if t[i] == 1:
            gradients.append(sofx[i]*(1-sofx[i]))
          else:
            gradients.append(-sofx[0]*sofx[i])
        return np.array(gradients)*soutput

class RowSum():

    def __init__(self):
      pass

    def forward(self, x):
      x = np.array(x)
      sumd = x.sum(axis=1)
      self.x = x.shape[1]
      return sumd

    def backward(self, gy):
      gy = np.array(gy)
      return np.broadcast_to(gy[:, None], (gy.shape[0], self.x))

class Expand():

    def __init__(self):
        pass

    def forward(self, size, x):
        return np.full(size, x)

    def backward(self, gy):
      gy = np.array(gy)
      return gy.sum(), None

# Input layer
inputs = np.array([1.,-1.])

# First layer
W = np.array([[1., 1., 1.], [-1., -1., -1.]])
b = np.array([0., 0., 0.])

# Second layer
V = np.array([[1., 1.], [-1., -1.], [-1., -1.]])
c = np.array([0., 0.])

# true class
t=np.array([1,0])

#activation function
sigmoid = Sigmoid()
softmax = Softmax()
rowsum = RowSum()
expand = Expand()

# Forward pass
k = np.dot(inputs, W) + b
h = sigmoid.forward(k)
s = np.dot(h, V) + c
o = softmax.forward(s)

# Cross-entropy loss derivative for correct class
l, dl = [(-np.log(o[i]), -1/o[i]) for i in range(len(o)) if t[i] == 1][0]

# Backward pass
ds = np.array(softmax.backward(dl)) #dl/ds c
dh = np.dot(ds, np.array(V).T) #dl/dh b
dk = sigmoid.backward(dh)# dl/dk 
dw = np.dot(np.array([inputs]).T, np.array([dk])) #dk/dw W
dv = np.dot(np.array([h]).T, np.array([ds]))

# Print the derivatives
print("W:", dw)
print("b", dh)
print("V", dv)
print("c", ds)

