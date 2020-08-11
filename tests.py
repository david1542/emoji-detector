import unittest
import numpy as np
from network_theta import Network
import utils

class TestNeuralNetwork(unittest.TestCase):    
    def test_feed_forward(self):
        X = np.round(np.cos(np.array([[1, 2], [3, 4], [5, 6]])), 4)
        y = np.array([4, 2, 3])
        y_onehot = utils.create_output_matrix(y, 4)
        
        num_labels = 4
        
        input_layer = 2
        hidden_layer = 2
        output_layer = num_labels
        layers = [input_layer, hidden_layer, output_layer]
        
        # Create a new neural network
        network = Network(layers)
        
        params = np.array(range(1,19)) / 10
        network.weights = params

        a, z = network.feed_forward(X)
        
        self.assertEqual(len(a), 3)
        self.assertEqual(len(z), 2)
        
        # Weighted inputs
        self.assertEqual(z[0].shape, (3, 2))
        self.assertEqual(z[1].shape, (3, 4))
        
        # Activations
        self.assertEqual(a[0].shape, (3, 3))
        self.assertEqual(a[1].shape, (3, 3))
        self.assertEqual(a[2].shape, (3, 4))

if __name__ == '__main__':
    unittest.main()