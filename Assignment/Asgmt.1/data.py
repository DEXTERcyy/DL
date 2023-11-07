# -- assignment 1 --
import numpy as np
from urllib import request
import gzip
import pickle
import os
import math

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

    if not os.path.isfile('mnist.pkl'):
        init()

    xtrain, ytrain, xtest, ytest = load()
    xtl, xsl = xtrain.shape[0], xtest.shape[0]

    if flatten:
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

#init=init()
softmax=Softmax()
sigmoid=Sigmoid()

(xtrain, ytrain), (xval, yval), num_cls = load_synth()

def initialize_weights(input_dim, hidden_dim, output_dim):
    w1 = np.random.randn(input_dim, hidden_dim) * 0.01
    b1 = np.zeros(hidden_dim)
    w2 = np.random.randn(hidden_dim, output_dim) * 0.01
    b2 = np.zeros(output_dim)
    return w1, b1, w2, b2

def compute_loss(y_true, y_pred):
    epsilon = 1e-7
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    loss = np.sum(-y_true * np.log(y_pred))
    return loss / y_true.shape[0] # normalize by number of instances

def train_network(x_train, y_train, input_dim, hidden_dim, output_dim, learning_rate=0.01, num_epochs=100):
    w1, b1, w2, b2 = initialize_weights(input_dim, hidden_dim, output_dim)

    for i in range(num_epochs):
        # Forward Pass x_train->z1-a1->z2-y_pred
            z1 = np.dot(x_train, w1) + b1
            a1 = sigmoid(z1)
            z2 = np.dot(a1, w2) + b2
            y_pred = softmax(z2)

            loss = compute_loss(y_train, y_pred)
            print(f'Epoch {i+1}/{num_epochs}, Loss: {loss}')

            # Backward Pass
            dz2 = y_pred - y_train # derivative of softmax
            dw2 = np.dot(a1.T, dz2) # derivative of weights
            db2 = np.sum(dz2, axis=0) # derivative of bias
            da1 = np.dot(dz2, w2.T) # derivative of activation
            dz1 = da1 * sigmoid.backward(a1) # derivative of Sigmoid
            dw1 = np.dot(x_train.T, dz1)
            db1 = np.sum(dz1, axis=0)

            # SGD weight updating
            w1 -= learning_rate * dw1
            b1 -= learning_rate * db1
            w2 -= learning_rate * dw2
            b2 -= learning_rate * db2

    return w1, b1, w2, b2
    b2 -= learning_rate * db2