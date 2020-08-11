# %% Imports
from utils import *
from network import Network
from labels import num_labels
import numpy as np

# %% Load total data
X_total, y_total, y_onehot_total = load_data('training')

# %% Load test data
X_test, y_test, y_onehot_test = load_data('testing')

# %% Load data evenly
min_items = find_min_items(y_total)
X, y, y_onehot = load_even_data('training', min_items)

# %% Display some images
plot_figures(X, y_onehot, 10, 2, 5)

# %% Create network

input_layer = 2304
hidden_layer = 20
output_layer = num_labels
layers = [input_layer, hidden_layer, output_layer]

# Create a new neural network
network = Network(layers)
# network.weights = np.load('faces_weights.npy')

# %% Train network
weights = network.train_cg(X, y_onehot, r12n = 1, iterations = 100)
network.weights = weights
np.save('faces_weights', network.weights)

# %% Check accuracy
_, _, training_accuracy = network.compute_accuracy(X, y)
_, _, test_accuracy = network.compute_accuracy(X_test, y_test)

print('Training set accuracy: ' + str(training_accuracy) + '%')
print('Test set accuracy: ' + str(test_accuracy) + '%')

# %% Plot confusion matrix
y_test_pred = network.predict(X)
plot_confusion_matrix(y, y_test_pred)

# %% Plot data distribution
plot_data_dist(y)