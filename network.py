import numpy as np
import utils
import scipy.optimize as op

class Network:
    def __init__(self, layers):
        self.layers = layers
        self.weights = self.generate_params()

    # Function for generating theta multidimensional matrix
    def generate_params(self):
        theta = []
        epsilon = 0.12
        for i in range(len(self.layers) - 1):
            current_layer_units = self.layers[i]
            next_layer_units = self.layers[i + 1]
            theta_i = np.multiply(
                np.random.rand(next_layer_units, current_layer_units + 1),
                2 * epsilon - epsilon
            )
            
            # Appending the params to the theta matrix
            theta.append(theta_i)
        
        return utils.unroll_theta(theta)
    
    def back_prop(self, params, X, y, r12n = 0):
        m = X.shape[0]
        X = np.matrix(X)
        y = np.matrix(y)
        
        # reshape the parameter array into parameter matrices for each layer
        theta1, theta2 = utils.roll_theta(params, self.layers)
        
        a1, z2, a2, z3, h = self.feed_forward(X, [theta1, theta2])

        # initializations
        J = 0
        delta1 = np.zeros(theta1.shape)  # (25, 401)
        delta2 = np.zeros(theta2.shape)  # (10, 26)
        
        # compute the cost
        for i in range(m):
            first_term = np.multiply(-y[i,:], np.log(h[i,:]))
            second_term = np.multiply((1 - y[i,:]), np.log(1 - h[i,:]))
            J += np.sum(first_term - second_term)
        
        J = J / m
        
        # add the cost regularization term
        J += (float(r12n) / (2 * m)) * (np.sum(np.power(theta1[:,1:], 2)) + np.sum(np.power(theta2[:,1:], 2)))
        
        # perform backpropagation
        for t in range(m):
            a1t = a1[t,:]  # (1, 401)
            z2t = z2[t,:]  # (1, 25)
            a2t = a2[t,:]  # (1, 26)
            ht = h[t,:]  # (1, 10)
            yt = y[t,:]  # (1, 10)
            
            d3t = ht - yt  # (1, 10)
            
            z2t = np.insert(z2t, 0, values=np.ones(1))  # (1, 26)
            d2t = np.multiply((theta2.T * d3t.T).T, utils.sigmoidGradient(z2t))  # (1, 26)
            
            delta1 = delta1 + (d2t[:,1:]).T * a1t
            delta2 = delta2 + d3t.T * a2t
            
        delta1 = delta1 / m
        delta2 = delta2 / m
        
        # add the gradient regularization term
        delta1[:,1:] = delta1[:,1:] + (theta1[:,1:] * r12n) / m
        delta2[:,1:] = delta2[:,1:] + (theta2[:,1:] * r12n) / m
        
        # unravel the gradient matrices into a single array
        grad = np.concatenate((np.ravel(delta1), np.ravel(delta2)))
        
        print(J)
        return J, grad

    # Function for training the neural network using conjugate gradient algorithm
    def train_cg(self, X, y, r12n = 0, iterations = 50):        
        options = {'maxiter': iterations}
        
        input_layer, hidden_layer, output_layer = self.layers
        result = op.minimize(fun = self.back_prop,
                             x0 = self.weights,
                             args=(X, y, r12n),
                             jac = True,
                             method = 'TNC',
                             options = options)        
        return result.x
        
    def feed_forward(self, X, weights):
        m = X.shape[0]
        
        theta1, theta2 = weights
        a1 = np.insert(X, 0, values=np.ones(m), axis=1)
        z2 = a1 * theta1.T
        a2 = np.insert(utils.sigmoid(z2), 0, values=np.ones(m), axis=1)
        z3 = a2 * theta2.T
        h = utils.sigmoid(z3)
        
        return a1, z2, a2, z3, h

    def predict(self, X):
        weights = utils.roll_theta(self.weights, self.layers)
        _, _, _, _, output = self.feed_forward(X, weights)

        # Finding the max index in the output layer
        return np.array(np.argmax(output, axis=1) + 1)
            
    def compute_accuracy(self, X, y):        
        predictions = self.predict(X)    
        intersection = [1 if a == b else 0 for (a, b) in zip(predictions, y)]
        
        correct = np.count_nonzero(intersection)
        incorrect = len(intersection) - correct
        accuracy = (sum(map(int, intersection)) / float(len(intersection))) * 100
        
        
        return correct, incorrect, accuracy 