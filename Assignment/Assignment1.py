import math

class Sigmoid():

  def __init__(self):
    pass

  def forward(self, x):
    return 1 / (1 + math.exp(-x))

  def backward(self, x, y):
    return y * (1 - y)

class Softmax():

    def __init__(self):
        pass

    def forward(self, x):
        return [math.exp(i) / sum([math.exp(j) for j in x]) for i in x]

    def backward(self, x, y):
        return y * (1 - y)

# Input layer
inputs = [1.,-1.]

# First layer
weights_layer_1 = [[1., 1., 1.], [-1., -1., -1.]]
biases_layer_1 = [0., 0., 0.]

# Second layer
weights_layer_2 = [[1., 1.], [-1., -1.], [-1., -1.]]
biases_layer_2 = [0., 0.]

# Sigmoid activation function
sigmoid = Sigmoid()

# Softmax activation function
softmax = Softmax()

# Forward pass
k_layer_1 = []
for i in range(len(biases_layer_1)):
  k_layer_1.append(sum([inputs[j] * weights_layer_1[j][i] for j in range(len(inputs))]) + biases_layer_1[i])

h_layer_1 = []
for i in range(len(biases_layer_1)):
  h_layer_1.append(sigmoid.forward(k_layer_1[i]))

k_layer_2 = []
for i in range(len(biases_layer_2)):
  k_layer_2.append(sum([h_layer_1[j] * weights_layer_2[j][i] for j in range(len(h_layer_1))]) + biases_layer_2[i])

p_layer_2=softmax.forward(k_layer_2)

# Loss
target_class = 0
loss = -math.log(p_layer_2[target_class])

# Backward pass
d_loss_dk_layer_2 = []
for i in range(2):
  d_loss_dk_layer_2.append(sigmoid.backward(k_layer_2[i], p_layer_2[target_class]))

d_loss_dw_layer_2 = []
for i in range(2):
  d_loss_dw_layer_2.append([])
  for j in range(len(h_layer_1)):
    d_loss_dw_layer_2[i].append(h_layer_1[j] * d_loss_dk_layer_2[i])

d_loss_db_layer_2 = d_loss_dk_layer_2

d_loss_dk_layer_1 = []
for i in range(2):
  d_loss_dk_layer_1.append(sum([d_loss_dw_layer_2[j][i] * d_loss_dk_layer_2[j] for j in range(len(d_loss_dw_layer_2))]) + d_loss_dk_layer_2[1] * d_loss_dk_layer_2[1])

d_loss_dw_layer_1 = []
for i in range(2):
  d_loss_dw_layer_1.append([])
  for j in range(len(inputs)):
    d_loss_dw_layer_1[i].append(inputs[j])

d_loss_db_layer_1 = d
