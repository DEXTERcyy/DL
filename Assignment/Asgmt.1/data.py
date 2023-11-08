# -- assignment 1 --
import numpy as np
from urllib import request
import gzip
import pickle
import os

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

def initialize_weights(input_dim, hidden_dim, output_dim):
    w1 = np.random.randn(input_dim, hidden_dim) * 0.01
    b1 = np.zeros(hidden_dim)
    w2 = np.random.randn(hidden_dim, output_dim) * 0.01
    b2 = np.zeros(output_dim)
    return np.array(w1), np.array(b1), np.array(w2),np.array(b2)

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

def train_network(x_train, y_train, input_dim, hidden_dim, output_dim, learning_rate=0.01, num_epochs=10):
    w1, b1, w2, b2 = initialize_weights(input_dim, hidden_dim, output_dim)
    print(f'Starting training for {num_epochs} epochs:')
    for i in range(num_epochs):
            print(f'Epoch {i+1}/{num_epochs}')
        # Forward Pass x_train->k-h->s-o
            k = np.dot(x_train, w1) + b1
            h = sigmoid.forward(k)
            s = np.dot(h, w2) + b2
            o = softmax.forward(s) #pred
            loss = compute_loss(o, y_train)
            print(f'Loss: {loss}')

            # Backward Pass
            db2 = np.array(softmax.backward(y_train)) #dl/ds c
            db1 = np.dot(db2, np.array(w2).T) #dl/dh b
            dk = sigmoid.backward(db1)# dl/dk 
            dw1 = np.dot(np.array([x_train]).T, np.array([dk])) #dk/dw W
            dw2 = np.dot(np.array([h]).T, np.array([db2]))

            # SGD weight updating
            w1 -= learning_rate * dw1
            b1 -= learning_rate * db1
            w2 -= learning_rate * dw2
            b2 -= learning_rate * db2

    return w1, b1, w2, b2



(xtrain, ytrain), (xval, yval), num_cls = load_synth()
#train_network(np.array(xtrain), np.array(ytrain), 2, 3, 1, learning_rate=0.01, num_epochs=100)
#train_network(np.array([[1,-1],[0.5,0.5]]), np.array([1,0]), 2, 3, 2, learning_rate=0.01, num_epochs=10)