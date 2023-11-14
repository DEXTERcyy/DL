import numpy as np

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
        sofx = np.exp(x) / sum([np.exp(j) for j in x])
        self.sofx = sofx
        return sofx

    def backward(self,t):
        sofx = self.sofx
        return np.array(sofx)-np.array(t)

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

def compute_loss(y_true, y_pred): # compute cross-entropy loss
    loss = 0
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    for i in range(y_true.shape[0]):
        loss += -y_true[i]*np.log(y_pred[i])
    return loss

# Input layer
inputs = [1.,-1.]

# First layer
W = [[1., 1., 1.], [-1., -1., -1.]]
b = [0., 0., 0.]

# Second layer
V = [[1., 1.], [-1., -1.], [-1., -1.]]
c = [0., 0.]

# true class
t=[1,0]

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
l=compute_loss(t,o)
print("Loss:", l)

# Backward pass
ds = np.array(softmax.backward(t)) #dl/ds c
dh = np.dot(ds, np.array(V).T) #dl/dh 
dk = sigmoid.backward(dh)# dl/dk b
dw = np.dot(np.array([inputs]).T, np.array([dk])) #dk/dw W
dv = np.dot(np.array([h]).T, np.array([ds])) #dl/dv V

# Print the derivatives
print("W:", dw)
print("b", dh)
print("V", dv)
print("c", ds)

