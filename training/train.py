### Add lines to import modules as needed
import tensorflow as tf
import numpy as np
import pickle
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
## 

def build_model():
  inputs = tf.keras.layers.Input(shape=(96,96,1))
  x = tf.keras.layers.SeparableConv2D(4, kernel_size=5,activation='relu',padding='same')(inputs)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.SeparableConv2D(8, kernel_size=5,activation='relu',padding='same')(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.SeparableConv2D(16, kernel_size=5,strides=(2,2),activation='relu',padding='same')(x)
  x = tf.keras.layers.BatchNormalization()(x)  
  x = tf.keras.layers.SeparableConv2D(16, kernel_size=3,activation='relu',padding='same')(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.SeparableConv2D(32, kernel_size=5,activation='relu', padding='same')(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.SeparableConv2D(32, kernel_size=3,strides=(2,2),activation='relu', padding='same')(x)
  x = tf.keras.layers.BatchNormalization()(x)
  res_con_1 = tf.keras.layers.Dropout(0.3)(x)
  
  x = tf.keras.layers.SeparableConv2D(64, kernel_size=5,activation='relu',padding='same')(res_con_1)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.SeparableConv2D(64, kernel_size=3,strides=(2,2),activation='relu',padding='same')(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.SeparableConv2D(64, kernel_size=3,activation='relu',padding='same')(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Dropout(0.3)(x)
  res_con_1 = tf.keras.layers.SeparableConv2D(64, kernel_size=(1,1),strides=(2,2))(res_con_1)
  res_con_2 = tf.keras.layers.add([x, res_con_1])

  x = tf.keras.layers.SeparableConv2D(128, kernel_size=5,activation='relu',padding='same')(res_con_2)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.SeparableConv2D(128, kernel_size=3,strides=(2,2),activation='relu',padding='same')(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.SeparableConv2D(256, kernel_size=3, activation='relu',padding='same')(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Dropout(0.3)(x)

  x = tf.keras.layers.MaxPool2D(pool_size = (2,2), strides = (2,2))(x)
  x = tf.keras.layers.Flatten()(x)
  x = tf.keras.layers.Dense(16)(x)
  x = tf.keras.layers.Dropout(0.3)(x)
  outputs = tf.keras.layers.Dense(2)(x)

  model = tf.keras.Model(inputs, outputs) # Add code to define model 1.
  return model

if __name__ == '__main__':

    file = "cat_dataset.pkl"

    with open(file, "rb") as f:
       train_images, train_labels = pickle.load(f)

    train_images, val_images, train_labels, val_labels = train_test_split(
    train_images, train_labels, test_size=0.2, random_state=42
    )

    model = build_model()
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    train_hist = model.fit(train_images, train_labels, 
                         validation_data=(val_images, val_labels),
                         epochs=50)


    plt.subplot(2,1,1)
    plt.plot(train_hist.epoch, train_hist.history['accuracy'], train_hist.epoch, train_hist.history['val_accuracy'])
    plt.legend(['Accuracy', 'Validation Acc'])
    plt.subplot(2,1,2)
    plt.plot(train_hist.epoch, train_hist.history['loss'], train_hist.history['val_loss'])
    plt.legend(['Loss', 'Val Loss'])

    pred_probs = model.predict(val_images)  # Get probability outputs
    pred_labels = np.argmax(pred_probs, axis=1)  # Convert to class labels

    # Compute confusion matrix
    cm = confusion_matrix(val_labels, pred_labels)

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(val_labels), yticklabels=np.unique(val_labels))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

    model.save('test_model.h5')
