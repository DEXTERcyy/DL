# -- assignment 1 --
import numpy as np
import random
from urllib import request
import gzip
import pickle
import os
import matplotlib.pyplot as plt

class Softmax():

    def __init__(self):
        self.sofx = None

    def forward(self, x):
        sofx = [np.exp(i) / sum([np.exp(j) for j in x]) for i in x]
        self.sofx = sofx
        return self.sofx

    def backward(self,t):
        sofx = self.sofx
        return np.array(sofx)-t

class Sigmoid():

  def __init__(self):
    self.sigx = None

  def forward(self, x):
    x = np.clip(x, -500, 500)  # limit values to avoid overflow in exp
    sigx = 1 / (1 + np.exp(-x))
    self.sigx = sigx
    return sigx

  def backward(self, goutput):
    sigx = self.sigx
    return goutput * sigx * (1 - sigx)

def load_synth(num_train=60_000, num_val=10_000, seed=0):
    """
    Load some very basic synthetic data that should be easy to classify. Two features, so that we can plot the
    decision boundary (which is an ellipse in the feature space).

    :param num_train: Number of training instances
    :param num_val: Number of test/validation instances
    :param num_features: Number of features per instance

    :return: Two tuples and an integer: (xtrain, ytrain), (xval, yval), num_cls. The first contains a matrix of training
     data with 2 features as a numpy floating point array, and the corresponding classification labels as a numpy
     integer array. The second contains the test/validation data in the same format. The last integer contains the
     number of classes (this is always 2 for this function).
    """
    np.random.seed(seed) # for reproducibility

    THRESHOLD = 0.6 # threshold for the quadratic form
    quad = np.asarray([[1, -0.05], [1, .4]]) # quadratic form matrix

    ntotal = num_train + num_val # total number of instances

    x = np.random.randn(ntotal, 2) # generate random data

    # compute the quadratic form
    q = np.einsum('bf, fk, bk -> b', x, quad, x) # einsum is Einstein summation [ntotal,]
    y = (q > THRESHOLD).astype(np.int64)

    return (x[:num_train, :], y[:num_train]), (x[num_train:, :], y[num_train:]), 2

def load_mnist(final=False, flatten=True):
    """
    Load the MNIST data.

    :param final: If true, return the canonical test/train split. If false, split some validation data from the training
       data and keep the test data hidden.
    :param flatten: If true, each instance is flattened into a vector, so that the data is returns as a matrix with 768
        columns. If false, the data is returned as a 3-tensor preserving each image as a matrix.

    :return: Two tuples and an integer: (xtrain, ytrain), (xval, yval), num_cls. The first contains a matrix of training
     data and the corresponding classification labels as a numpy integer array. The second contains the test/validation
     data in the same format. The last integer contains the number of classes (this is always 2 for this function).

     """

    if not os.path.isfile('mnist.pkl'): # download MNIST dataset if not yet done
        init()

    xtrain, ytrain, xtest, ytest = load() # load data
    xtl, xsl = xtrain.shape[0], xtest.shape[0] # number of training and test instances

    if flatten: # flatten images
        xtrain = xtrain.reshape(xtl, -1)    
        xtest  = xtest.reshape(xsl, -1) 

    if not final: # return the flattened images
        return (xtrain[:-5000], ytrain[:-5000]), (xtrain[-5000:], ytrain[-5000:]), 10

    return (xtrain, ytrain), (xtest, ytest), 10

# Numpy-only MNIST loader. Courtesy of Hyeonseok Jung
# https://github.com/hsjeong5/MNIST-for-Numpy

filename = [
["training_images","train-images-idx3-ubyte.gz"],
["test_images","t10k-images-idx3-ubyte.gz"],
["training_labels","train-labels-idx1-ubyte.gz"],
["test_labels","t10k-labels-idx1-ubyte.gz"]
]

def download_mnist():
    base_url = "http://yann.lecun.com/exdb/mnist/"
    for name in filename:
        print("Downloading "+name[1]+"...")
        request.urlretrieve(base_url+name[1], name[1])
    print("Download complete.")

def save_mnist():
    mnist = {}
    for name in filename[:2]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28*28)
    for name in filename[-2:]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
    with open("mnist.pkl", 'wb') as f:
        pickle.dump(mnist,f)
    print("Save complete.")

def init():
    download_mnist()
    save_mnist()

def load():
    with open("mnist.pkl",'rb') as f:
        mnist = pickle.load(f)
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]

def initialize_weights(input_dim, hidden_dim, output_dim):
    w1 = np.random.randn(input_dim, hidden_dim) * 0.01
    b1 = np.zeros(hidden_dim)
    w2 = np.random.randn(hidden_dim, output_dim) * 0.01
    b2 = np.zeros(output_dim)
    return w1, b1, w2, b2

def compute_loss(y_true, y_pred): # compute cross-entropy loss
    loss = 0
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    eps = 1e-10 # small constant to avoid numerical instability
    for i in range(y_true.shape[0]):
        loss += -y_true[i]*np.log(y_pred[i]+eps)
    return loss

#init=init()
softmax=Softmax()
sigmoid=Sigmoid()

def train_network(xtrain, ytrain, input_dim, hidden_dim, output_dim, learning_rate=0.01, num_epochs=10):
    w1, b1, w2, b2 = initialize_weights(input_dim, hidden_dim, output_dim)
    print(f'Starting training for {num_epochs} epochs:')
    xtrain_np = np.array(xtrain)
    # Get the min and max for each column
    min_vals = xtrain_np.min(axis=0)
    max_vals = xtrain_np.max(axis=0)
    # Normalize to the range [-1, 1]
    xtrain = (xtrain_np - min_vals) / (max_vals - min_vals)
    for i in range(num_epochs):
            print(f'Epoch {i+1}/{num_epochs}')
            #randomly choose a training example j from xtrain
            j = random.randint(0, len(xtrain) - 1)
            x_train, y_train = xtrain[j], np.eye(2)[ytrain][j]
            # Forward Pass x_train->k-h->s-o
            k = np.dot(x_train, w1) + b1
            h = sigmoid.forward(k)
            s = np.dot(h, w2) + b2
            o = softmax.forward(s) #pred
            loss = compute_loss(o, y_train)
            print(f'Loss: {loss}')

            # Backward Pass
            db2 = np.array(softmax.backward(y_train)) #dl/ds c
            db1 = np.dot(db2, np.array(w2).T) #dl/dh
            dk = sigmoid.backward(db1)# dl/dk b
            dw1 = np.dot(np.array([x_train]).T, np.array([dk])) #dk/dw W
            dw2 = np.dot(np.array([h]).T, np.array([db2]))

            # SGD weight updating
            w1 -= learning_rate * dw1
            b1 -= learning_rate * db1
            w2 -= learning_rate * dw2
            b2 -= learning_rate * db2
    print("w1:\n", w1)
    print("b1:\n", b1)
    print("w2:\n", w2)
    print("b2:\n", b2)
    print("final loss:\n", loss)
    return w1, b1, w2, b2, loss
(xtrain, ytrain), (xval, yval), num_cls = load_synth()
train_network(np.array(xtrain), np.array(ytrain), 2, 3, 2, learning_rate=0.01, num_epochs=100)

def train_MNIST_network(xtrain, ytrain, input_dim, hidden_dim, output_dim, learning_rate=0.01, num_epochs=10, batch_size=100):
    w1, b1, w2, b2 = initialize_weights(input_dim, hidden_dim, output_dim)
    # Normalize
    x_train = xtrain.astype('float64')
    x_train /= np.max(x_train)
    for i in range(num_epochs):
        for j in range(0, len(xtrain), batch_size):
            dw1, db1, dw2, db2, loss = 0, 0, 0, 0, 0
            for k in range(j, j + batch_size):
                print(f'Epoch:{i+1}/{num_epochs},Batch:{j},Train:{k}.')
                x_train, y_train = xtrain[k], np.eye(10)[ytrain][k]
                # Forward Pass
                k = np.dot(x_train, w1) + b1
                h = sigmoid.forward(k)
                s = np.dot(h, w2) + b2
                o = softmax.forward(s)
                # Compute Loss
                loss += compute_loss(y_train, o)
                # Backward Pass
                db2 += np.array(softmax.backward(y_train)) #dl/ds c
                db1 += np.dot(db2, np.array(w2).T) #dl/dh
                dk = sigmoid.backward(db1)# dl/dk b
                dw1 += np.dot(np.array([x_train]).T, np.array([dk])) #dk/dw W
                dw2 += np.dot(np.array([h]).T, np.array([db2]))
                # Update Weights
            w1 -= learning_rate * dw1/batch_size
            b1 -= learning_rate * db1/batch_size
            w2 -= learning_rate * dw2/batch_size
            b2 -= learning_rate * db2/batch_size
            loss /= batch_size
            print(f'Batch loss: {loss}')

    return w1, b1, w2, b2
(xtrain, ytrain), (xval, yval), num_cls = load_mnist()
train_MNIST_network(np.array(xtrain), np.array(ytrain), 784, 300, 10, learning_rate=0.01, num_epochs=100)
print("Done")



