import numpy as np

class Dense():
  def __init__(self, inputs, neurons):
    self.weights = 0.10 * np.random.randn(inputs, neurons)
    self.biases = np.zeros((1, neurons))

  # forward propegation
  def forward(self, inputs):
    self.inputs = inputs
    self.output = np.dot(inputs, self.weights) + self.biases

  # backward propegation
  def backward(self, dvalues):
    self.dinputs = np.dot(dvalues, self.weights.T)
    self.dweights = np.dot(self.inputs.T, dvalues)
    self.dbiases = np.sum(dvalues, axis=0, keepdims=True)