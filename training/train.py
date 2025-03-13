### Add lines to import modules as needed
import tensorflow as tf
import numpy as np
import pickle
## 

def build_model():
  inputs = tf.keras.layers.Input(shape=(240,240,3))
  x = tf.keras.layers.SeparableConv2D(32, kernel_size=5,activation='relu', padding='same')(inputs)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.SeparableConv2D(64, kernel_size=3,strides=(2,2),activation='relu', padding='same')(x)
  x = tf.keras.layers.BatchNormalization()(x)
  res_con_1 = tf.keras.layers.Dropout(0.3)(x)
  
  x = tf.keras.layers.SeparableConv2D(64, kernel_size=3,strides=(2,2),activation='relu',padding='same')(res_con_1)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.SeparableConv2D(64, kernel_size=3,strides=(2,2),activation='relu',padding='same')(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.SeparableConv2D(64, kernel_size=3,activation='relu',padding='same')(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Dropout(0.3)(x)
  res_con_1 = tf.keras.layers.SeparableConv2D(64, kernel_size=(1,1),strides=(4,4))(res_con_1)
  res_con_2 = tf.keras.layers.add([x, res_con_1])

  x = tf.keras.layers.SeparableConv2D(128, kernel_size=3,strides=(2,2),activation='relu',padding='same')(res_con_2)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Dropout(0.3)(x)
  res_con_2 = tf.keras.layers.SeparableConv2D(128, kernel_size=(1,1),strides=(2,2))(res_con_2)

  x = tf.keras.layers.MaxPool2D(pool_size = (2,2), strides = (2,2))(res_con_2)
  x = tf.keras.layers.Flatten()(x)
  x = tf.keras.layers.Dense(128)(x)
  outputs = tf.keras.layers.Dense(10)(x)

  model = tf.keras.Model(inputs, outputs) # Add code to define model 1.
  return model

if __name__ == '__main__':

    file = "cat_dataset.pkl"

    with open(file, "rb") as f:
       train_images, train_labels = pickle.load(f)


    model = build_model()
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    train_loss = model.fit(train_images, train_labels, epochs=50)

    #test_loss, test_accuracy = model.evaluate(test_images, test_labels)
    model.save('test_model.h5')