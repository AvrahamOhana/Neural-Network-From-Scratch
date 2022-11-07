import numpy as np

class ReLU():

  # Forward propegation
  def forward(self, inputs):
    self.inputs = inputs
    self.output = np.maximum(0, inputs)

  # backward propegation
  def backward(self, dvalues):

    self.dinputs = dvalues.copy()
    # zero the gradient where inputs values are negative
    self.dinputs[self.inputs <= 0] = 0
    
class Softmax():
  def forward(self, inputs):
    # do un-normalize process to the inputs
    exp = np.exp(inputs - np.max(inputs, axis=1,keepdims=True))

    # calculate the probabilities
    P = exp / np.sum(exp, axis=1,keepdims=True)
    self.output = P