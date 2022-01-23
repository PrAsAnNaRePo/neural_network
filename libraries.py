

'''

SIMPLE NEURAL NETWORKS LAYER IMPLEMENTATION
** only numpy, no tensorflow, keras, pytorch **

prerequisite:
    * you should know how these node are working.!
    * matrics calculus
    * partial differentiation


 AND EACH FUNCTION HAS THE BACK PROPAGATION PROPERTY..!

 sorry back propagation can't explain by text
'''


import random
import numpy as np

x = [1, 2, 3, 4, 5] #inputs
y = [2, 4, 6, 8, 10] #outputs


# creating a dense layer




class Dense:
    def __init__(self, n_inputs, n_nodes):# setting the initial params
        self.n_nodes = n_nodes
        self.output = None
        self.weights = 0.1 * np.random.rand(n_inputs, n_nodes)# initially the weights nad bias are set to some values
        self.bias = np.random.rand(1, n_nodes)
        self.n_weights = range(n_inputs)

    def forward_propagation(self, inputs): # BY the formula sum((Wi * Ii)) + bias
        self.inputs = inputs
        print(f'input recieved : {inputs}')
        print(f'assigned weights : {self.weights} caz no.nodes : {self.n_nodes}')
        self.output = np.dot(self.inputs, self.weights) + self.bias
    def back_prop(self):
        self.e = self.inputs

# creating a relu activation function...
# refer this image https://cdn-images-1.medium.com/max/1600/1*DfMRHwxY1gyyDmrIAd-gjQ.png

class Relu:
    def __init__(self):
        self.output = None

    def forward(self, inputz): # if the input is <= 0 --> 0
        self.inputz = inputz
        self.out = []
        self.out.append(np.maximum(0.01 * inputz, inputz))
        self.output = self.out.copy()
    def back_prop(self):
        self.e = 1 if self.inputz > 0 else 0


# Softmax activation function...
class Softmax:
    def __init__(self):
        self.output = None
        self.probabilities = None
        self.exp_values = None

    def forward(self, inputs):
        self.exp_values = np.exp(inputs - np.max(inputs, axis=1,
                                                 keepdims=True))
        # Normalize them for each sample
        self.probabilities = self.exp_values / np.sum(self.exp_values, axis=1,
                                                      keepdims=True)
        self.output = self.probabilities

# Sigmoid activation function....

class Sigmoid:
    def __init__(self):
        self.output = None

    def forward(self, x):
        self.output = 1 / (1 + np.exp(-x))


loss = 'mse'
class MSE:
    def __init__(self, labels, output):
        self.total_loss = 0
        self.labels = labels
        self.outputs = output
        self.n_labels = len(labels)
    def find_loss(self):  # mse = 1/N (true_value - predicted_value) ** 2
        self.individual_loss = []
        for i in range(self.n_labels):
            self.l = (1 / self.n_labels) * ( float(self.labels[i]) - float(self.outputs[i]) )
            self.individual_loss.append(self.l)
        if len(self.individual_loss) == 1:
            self.total_loss = self.individual_loss
        else:
            self.total_loss = sum(self.individual_loss)
    def back_prop(self):
        self.e = 0
        for i in range(self.n_labels):
            self.e = 2 * (1/self.n_labels) * (self.labels[i] - self.outputs[i]) * (0-self.outputs[i])

