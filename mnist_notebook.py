from scipy.io import loadmat
import numpy as np
from network import Network
import utils

# %% Load data
data = loadmat('ex3data1.mat')

X = data["X"]
y = data["y"]
y_onehot = utils.create_output_matrix(y, 10)

# %% Create network
num_labels = 10

input_layer = 400
hidden_layer = 25
output_layer = num_labels
layers = [input_layer, hidden_layer, output_layer]

# Create a new neural network
network = Network(layers)
network.weights = np.load('mnist_weights.npy')

# %% Traing network
network.train_cg(X, y_onehot, r12n = 0, iterations = 2)
np.save('mnist_weights', network.weights)

# %% Check accuracy
correct, incorrect, accuracy = network.compute_accuracy(X, y)
print('Training set accuracy: ' + str(accuracy) + '%')

# %% Confusion matrix
y_pred = network.predict(X) 
utils.plot_confusion_matrix(y, y_pred)