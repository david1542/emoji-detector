import re
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from labels import emotions, num_labels

def load_data(t):
    X = np.load('data/' + t + '/images.npy').astype(float)
    y = np.load('data/' + t + '/outputs.npy').astype(int)
    
    mean, std = X.mean(), X.std()
    X = (X - mean) / std

    y_onehot = create_output_matrix(y, num_labels)
    
    return X, y, y_onehot

def load_even_data(t, amount):
    X, y, _ = load_data(t)
    
    m = X.shape[1]
    X_even = np.empty([0, m])
    y_even = np.array([])

    for label in range(1, num_labels + 1):
        indices = (np.where(y == label)[0])[:amount]
        X_even = np.append(X_even, X[indices, :], axis=0)
        y_even = np.append(y_even, y[indices], axis=0)

    y_even = y_even.astype(int)
    y_onehot_even = create_output_matrix(y_even, num_labels)
    return X_even, y_even, y_onehot_even

def find_min_items(y):
    lengths = [len(np.where(y == label)[0]) for label in range(1, num_labels + 1)]
    print(lengths)
    return min(lengths)

def create_output_vector(label, num_labels):
    return np.concatenate([np.zeros(label - 1), [1], np.zeros(num_labels - label)]).astype(int)

def create_output_matrix(labels, num_labels):
    outputs = np.zeros((len(labels), num_labels))
    m = len(labels)
    
    for i in range(m):
        outputs[i] = create_output_vector(labels[i], num_labels)
    
    return outputs
        
def sigmoid(z):
    return np.divide(1, np.add(1, np.exp(-z)))

def sigmoidGradient(z):
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))

def plot_figures(data, outputs, images_num, nrows, ncols):
    m = len(data)
    
    rand_indexes = random.sample(range(m), images_num)
    
    random_images = [data[i, :] for i in rand_indexes]
    random_outputs = [outputs[i] for i in rand_indexes]
    
    figures = []
    for i in range(images_num):
        figure = {
            "title": emotions[np.argmax(random_outputs[i], axis=0) + 1],
            "image": random_images[i].reshape((48, 48))
        }
                
        # Append figure to the figures collection
        figures.append(figure)

    if (images_num == 1):
        plt.imshow(figures[0]["image"], cmap=plt.gray())
        plt.title(figures[0]["title"])
    else:
        fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows)
        for ind,figure in enumerate(figures):
            axeslist.ravel()[ind].imshow(figure["image"], cmap=plt.gray())
            axeslist.ravel()[ind].set_title(figure["title"])
            axeslist.ravel()[ind].set_axis_off()

        plt.tight_layout() # optional

def sorted_alphanumeric(l):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key = alphanum_key)

def unroll_theta(theta):
    unrolled_theta = np.array([])
    for theta_i in theta:
        unrolled_theta = np.concatenate((unrolled_theta, theta_i.flatten(order='F').T), axis=0)
    
    return unrolled_theta

def roll_theta(params, layers):
    input_size, hidden_size, num_labels = layers
    
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

    return [theta1, theta2]

def unison_shuffle(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def plot_confusion_matrix(y_true, y_pred):
    labels = list(emotions.values())
    cm = confusion_matrix(y_true, y_pred)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm, cmap = plt.cm.Blues)
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.show()
    
def plot_data_dist(y):
    # Data to plot
    labels = list(emotions.values())
    sizes = []
    
    for i, label in enumerate(labels):
        count = (y == (i + 1)).sum()
        sizes.append(count)

    colors = ['blue', 'red', 'green', 'gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
    
    # Plot
    plt.pie(sizes, labels=labels, colors=colors,
    autopct='%1.1f%%', shadow=True, startangle=140)
    
    plt.axis('equal')
    plt.show()