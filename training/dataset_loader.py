import os
import numpy as np
import tensorflow as tf
#from tensorflow.keras.utils import img_to_array, load_img
import pickle  # To save the dataset

# Define paths
CAT_DIR = "dataset/cats"
NO_CAT_DIR = "dataset/no_cats"
IMAGE_SIZE = (240, 240)  # Resize images to 128x128

# Function to load and process images
def load_images_from_folder(folder, label):
    images = []
    labels = []
    
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            img = tf.keras.utils.load_img(file_path, target_size=IMAGE_SIZE)  # Load and resize
            img_array = tf.keras.utils.img_to_array(img) / 255.0  # Normalize pixel values
            images.append(img_array)
            labels.append(label)
        except Exception as e:
            print(f"Error loading image {file_path}: {e}")
    
    return images, labels

# Load images from both folders
cat_images, cat_labels = load_images_from_folder(CAT_DIR, 1)  # Label 1 for "cat"
no_cat_images, no_cat_labels = load_images_from_folder(NO_CAT_DIR, 0)  # Label 0 for "no cat"

# Combine dataset
X = np.array(cat_images + no_cat_images)
y = np.array(cat_labels + no_cat_labels)

# Shuffle dataset
indices = np.arange(len(X))
np.random.shuffle(indices)
X, y = X[indices], y[indices]

# Save dataset as a pickle file
with open("cat_dataset.pkl", "wb") as f:
    pickle.dump((X, y), f)

print(f"Dataset saved: {len(X)} images, Labels: {set(y)}")