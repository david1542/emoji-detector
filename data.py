from PIL import Image
from utils import plot_figures, sorted_alphanumeric
import numpy as np
import os, os.path

image_width = 48
image_height = 48
image_size = image_width * image_height

def import_images_as_pixels(path):
    images = []
    files = sorted_alphanumeric(os.listdir(path))

    for f in files:
        image = Image.open(os.path.join(path, f)).convert('L')        
        images.append(np.asarray(image).flatten())
    
    return np.array(images)

def load_images(subfolder):
    root_path = 'images/' + subfolder
    folders = [
        'Angry',
        'Fear',
        'Happy',
        'Neutral',
        'Sad',
        'Surprise'
    ]

    all_images = np.empty([0, image_size])
    all_outputs = np.empty([0])
    for i, f in enumerate(folders):
        path = root_path + '/' + f

        print('Importing images from ' + path)
        matrix = import_images_as_pixels(path)
        
        all_outputs = np.concatenate((all_outputs, np.full(len(matrix), i + 1)))
        all_images = np.concatenate((all_images, matrix), axis=0)
    
    data = {
        "images": all_images,
        "outputs": all_outputs 
    }
    return data