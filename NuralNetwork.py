import numpy as np
# import math
import nnfs
from nnfs.datasets import spiral_data  # code: https://gist.github.com/Sentdex/454cb20ec5acf0e76ee8ab8448e6266c

nnfs.init()
''''
input1 = [1, 2, 3, 2.5]
weight1 = [0.2, 0.8, -0.5, 1.0]
weight2 = [0.5, -0.91, 0.26, -0.5]
weight3 = [-0.26, -0.27, 0.17, 0.87]
bias1 = 2
bias2 = 3
bias3 = 0.5

output = [input1[0] * weight1[0] + input1[1] * weight1[1] + input1[2] * weight1[2] + input1[3] * weight1[3] + bias1,
          input1[0] * weight2[0] + input1[1] * weight2[1] + input1[2] * weight2[2] + input1[3] * weight2[3] + bias2,
          input1[0] * weight3[0] + input1[1] * weight3[1] + input1[2] * weight3[2] + input1[3] * weight3[3] + bias3]
print(output)

weights = [[0.2, 0.8, -0.5, 1.0], [0.5, -0.91, 0.26, -0.5], [-0.26, -0.27, 0.17, 0.87]]
biases = [2, 3, 0.5]
output = []
for neuron_weight, neuron_bias in zip(weights, biases):
    neuron_opt = 0
    for n_input, weight in zip(input1, neuron_weight):
        neuron_opt += n_input * weight
    neuron_opt += neuron_bias
    output.append(neuron_opt)

print(output)

weights = [[0.2, 0.8, -0.5, 1.0], [0.5, -0.91, 0.26, -0.5], [-0.26, -0.27, 0.17, 0.87]]
biases = [2, 3, 0.5]
output = np.dot(weights, input1) + biases
print(output)


input1 = [[1, 2, 3, 2.5], [2.0, 5.0, -1.0, 2.0], [-1.5, 2.7, 3.3, -0.8]]
weights1 = [[0.2, 0.8, -0.5, 1.0], [0.5, -0.91, 0.26, -0.5], [-0.26, -0.27, 0.17, 0.87]]
biases1 = [2, 3, 0.5]
weights2 = [[0.1, -0.14, 0.5], [-0.5, 0.12, -0.33], [-0.44, 0.73, -0.13]]
biases2 = [-1, 2, -0.5]
layer1_output = np.dot(input1, np.array(weights1).T) + biases1
layer2_output = np.dot(layer1_output, np.array(weights2).T) + biases2
print(layer1_output)

# X = [[1, 2, 3, 2.5], [2.0, 5.0, -1.0, 2.0], [-1.5, 2.7, 3.3, -0.8]]

E = math.e
lst = [4.8, 1.21, 2.385]
# exp_values = [E ** i for i in lst]
# print(exp_values)
# base = sum(exp_values)
# norm_values = [j / base for j in exp_values]
exp_values = np.exp(lst)
print(exp_values)
norm_values = exp_values / np.sum(exp_values)
print(norm_values)
print(sum(norm_values))
lst = [[4.8, 1.21, 2.385], [8.9, -1.81, 0.2], [1.41, 1.051, 0.026]]
exp_val = np.exp(lst)
norm_val = exp_val / np.sum(exp_val, axis=1, keepdims=True)
print(norm_val)
'''''
X, y = spiral_data(100, 3)


class Layer_Dense:
    output: None

    def __init__(self, n_input, n_neuron):
        self.weight = 0.10 * np.random.randn(n_input, n_neuron)
        self.bias = np.zeros((1, n_neuron))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weight) + self.bias


class Activation_Relu:
    output: None

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class Activation_Softmax:
    output: None

    def forward(self, inputs):
        exp_val = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_val / np.sum(exp_val, axis=1, keepdims=True)
        self.output = probabilities


# layer1 = Layer_Dense(2, 5)
# activation1 = Activation_Relu()
# layer2 = Layer_Dense(5, 2)
# layer1.forward(X)
# activation1.forward(layer1.output)
# print(layer1.output)
# print(activation1.output)

dense1 = Layer_Dense(2, 3)
activation1 = Activation_Relu()
dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()
dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
print(activation2.output[:10])
