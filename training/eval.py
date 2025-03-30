### Add lines to import modules as needed
import tensorflow as tf
import numpy as np
import pickle


if __name__ == '__main__':

   file = "test_dataset.pkl"

   with open(file, "rb") as f:
      test_images, test_labels = pickle.load(f)

   model = tf.keras.models.load_model("test_model.h5")
   test_loss, test_accuracy = model.evaluate(test_images, test_labels)
   model.summary()

   print(test_loss)
   print(test_accuracy)