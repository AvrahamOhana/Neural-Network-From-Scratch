import numpy as np
from activation import Softmax

# basic loss class. calculates the mean loss. 
class Loss():
    def calculate(self, output, y):
        # Calculate the samples losses
        loss_samples = self.forward(output, y)

        # Calculate the mean loss
        loss_data = np.mean(loss_samples)

        # Return the calculated data loss
        return loss_data
    
class CategoricalCrossentropy(Loss):
    # Forward pass
    def forward(self, y_pred, y_true):

        # Get the num of samples in single batch
        samples = len(y_pred)

        # Clipping the data to prevent division by 0
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

     
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
       
        # calculate the losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    # backward propegation
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])

        # convert the labels to one-hot labels
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        
        # calculate the deriviate inputs
        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples
        
class Softmax_Loss_CategoricalCrossentropy():
    # Creates activation and loss function objects
    def __init__(self):
        self.activation = Softmax()
        self.loss = CategoricalCrossentropy()

    # Forward propegation
    def forward(self, inputs, y_true):
        # the output layer activation function
        self.activation.forward(inputs)
        # Set the output
        self.output = self.activation.output
        # Calculate and return loss value
        return self.loss.calculate(self.output, y_true)

    # Backward propegation
    def backward(self, dvalues, y_true):
        # get the num of samples
        samples = len(dvalues)

        # convert labels to discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        # copy the dvalues
        self.dinputs = dvalues.copy()

        # calculate the gradient
        self.dinputs[range(samples), y_true] -= 1

        # normalize the gradient
        self.dinputs = self.dinputs / samples