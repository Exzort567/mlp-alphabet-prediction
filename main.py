import numpy as np
from joblib import dump, load
import os

class MLP:
    def __init__(self, inputSize, noOfHidden, noOfOutput, initial_hidden_bias=None, initial_output_bias=None):
        if os.path.exists('model_weights.joblib'):
            # Load weights and biases from file
            self.hW, self.oW, self.hB, self.oB = load('model_weights.joblib')
        else:
            # Initialize weights and biases randomly
            self.hW = np.random.randn(inputSize, noOfHidden)
            self.oW = np.random.randn(noOfHidden, noOfOutput)
            if initial_hidden_bias is not None:
                self.hB = initial_hidden_bias
            else:
                self.hB = np.random.randn(noOfHidden)

            if initial_output_bias is not None:
                self.oB = initial_output_bias
            else:
                self.oB = np.random.randn(noOfOutput)
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def forward(self, x):
        # Forward pass for the hidden layer
        hOutput = self.sigmoid(np.dot(x, self.hW) + self.hB)
        # Forward pass for the output layer
        yOutput = self.sigmoid(np.dot(hOutput, self.oW) + self.oB)
        return yOutput

    def backward(self, y, x):
        error_o = y * (1 - y)
        error_h = error_o * self.oW * (1 - self.sigmoid(np.dot(x, self.hW) + self.hB))
        return error_h, error_o
    
    def updateHidden(self, learning_rate, gradients, x):
        self.hW += learning_rate * np.outer(x, gradients)
        self.hB += learning_rate * gradients

    def updateOutput(self, learning_rate, gradients, yH):
        self.oW += learning_rate * np.outer(yH, gradients)
        self.oB += learning_rate * gradients
    
    def predict(self, x):
        # Perform forward pass
        yPredict = self.forward(x)
        
        
        if yPredict < 0.1:
            return 0
        else:
            return 1
            
    
    def fit(self, X, Y, learning_rate, max_epoch, threshold, weights_file='model_weights.joblib'):
        print("Initializing weights...")
        print("Training model with LR:", learning_rate, "Max Epoch:", max_epoch, "Threshold:", threshold)

        for epoch in range(1, max_epoch + 1):
            total_loss = 0

            for x, y in zip(X, Y):
                yPredict = self.forward(x) 
                error_o = yPredict * (1 - yPredict) * (y - yPredict)
                error_h = np.dot(error_o, self.oW.T) * (self.sigmoid(np.dot(x, self.hW) + self.hB) + (1 - self.sigmoid(np.dot(x, self.hW) + self.hB)))
                self.updateHidden(learning_rate, error_h, x)
                self.updateOutput(learning_rate, error_o, self.sigmoid(np.dot(x, self.hW) + self.hB))
                total_loss += 0.5 * np.sum((yPredict - y) ** 2)
            
            print("Epoch", epoch, "Error:", round(total_loss, 4))
            
            # Check for performance criteria
            if total_loss <= threshold:
                print("Error threshold has been reached. Training stopped.")
                # Save weights and biases to a file
                dump((self.hW, self.oW, self.hB, self.oB), weights_file)
            
                break
    
# Load dataset
dataset = np.loadtxt('alphabet.csv', delimiter=',')

# Extract input features (X) and target labels (Y)
X = dataset[:, :-1]  # Extract all columns except the last one as input features
Y = dataset[:, -1]   # Extract the last column as target labels

# Normalize input features (optional)
X /= X.max()

# Create MLP object
mlp = MLP(inputSize=35, noOfHidden=5, noOfOutput=1)

# Train the MLP
mlp.fit(X, Y, learning_rate=0.05, max_epoch=100000, threshold=0.001)
for x in X:
    print("Given the X:", ''.join(map(str, x)), "Prediction:", mlp.predict(x))

