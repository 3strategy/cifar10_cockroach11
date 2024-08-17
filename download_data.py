import tensorflow as tf
import numpy as np
import os
from PIL import Image

# Load the CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

# Define directories to save the images
train_dir = 'cifar10_train'
test_dir = 'cifar10_test'

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Save training images and labels
for i, (img, label) in enumerate(zip(train_images, train_labels)):
    label_dir = os.path.join(train_dir, str(label[0]))
    os.makedirs(label_dir, exist_ok=True)
    img = Image.fromarray(img)
    img.save(os.path.join(label_dir, f'{i}.png'))

# Save test images and labels
for i, (img, label) in enumerate(zip(test_images, test_labels)):
    label_dir = os.path.join(test_dir, str(label[0]))
    os.makedirs(label_dir, exist_ok=True)
    img = Image.fromarray(img)
    img.save(os.path.join(label_dir, f'{i}.png'))
