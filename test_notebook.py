# %% Imports
import numpy as np
from network_theta import Network
import utils

# %% Load data
X = np.round(np.cos(np.array([[1, 2], [3, 4], [5, 6]])), 4)
y = np.array([4, 2, 3])
y_onehot = utils.create_output_matrix(y, 4)

# %% Create network
num_labels = 4

input_layer = 2
hidden_layer = 2
output_layer = num_labels
layers = [input_layer, hidden_layer, output_layer]

# Create a new neural network
network = Network(layers)

params = np.array(range(1,19)) / 10
network.weights = params

# %% Traing network
network.train_cg(X, y_onehot, r12n = 0, iterations = 90)

# # %% Compute cost
# J = network.back_prop(params, X, y_onehot)
# print(J)
# %%  Test network accuracy on the training set
test_X = X
test_y = y

predictions = network.predict(test_X)
correct = [1 if a == b else 0 for (a, b) in zip(predictions, test_y)]
accuracy = (sum(map(int, correct)) / float(len(correct))) * 100

print('Testing set accuracy: ' + str(accuracy) + '%')

# %% Test network accuracy on the validation(test) set
# test_X = np.load('test_images.npy')
# test_y = np.load('test_outputs.npy').astype(int)

# predictions = network.predict(test_X.T)
# intersection = np.equal(predictions, test_y).astype(int)

# # Accuracy is measured by (correct predictions) / (total_predictions)
# accuracy = np.average(intersection) * 100
# print('Testing set accuracy: ' + str(accuracy) + '%')
