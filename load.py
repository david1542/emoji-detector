from data import load_images
import numpy as np

data = load_images('Training')
np.save('data/training/images', data['images'])
np.save('data/training/outputs', data['outputs'])

data = load_images('Testing')
np.save('data/testing/images', data['images'])
np.save('data/testing/outputs', data['outputs'])

