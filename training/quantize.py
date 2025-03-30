import tensorflow as tf
import numpy as np
import pickle

model = tf.keras.models.load_model('test_model.h5')
model.summary()


num_calibration_steps = 100
# converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
if True: 
  # If we omit this block, we'll get a floating-point TFLite model,
  # whose size in bytes should be ~4x the # parameters in the model
  # with this block, the weights and activations should be quantized to 8b integers, 
  # so the tflite file size (in bytes) will be ~ the number of model params, plus some overhead.

    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    file = "cat_dataset.pkl"
    with open(file, "rb") as f:
        train_images, train_labels = pickle.load(f)

    def representative_dataset_gen():
        for i in range(num_calibration_steps):
            next_input = train_images[i:i+1,:,:,:]
            yield [next_input] ## yield defines a generator

    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8] # use this one
    # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

    converter.inference_input_type = tf.int8  # or tf.uint8; should match dat_q in eval_quantized_model.py
    converter.inference_output_type = tf.int8  # or tf.uint8

    tflite_quant_model = converter.convert()

    tfl_file_name = "quant_model.tflite"
    with open(tfl_file_name, "wb") as fpo:
        fpo.write(tflite_quant_model)
    print(f"Wrote to {tfl_file_name}")