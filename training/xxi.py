import tensorflow as tf

# Load your quantized TFLite model
with open("quant_model.tflite", "rb") as f:
    tflite_model = f.read()

# Convert it to a C array
with open("model_quantized.h", "w") as f:
    f.write('#ifndef MODEL_QUANTIZED_H\n#define MODEL_QUANTIZED_H\n\n')
    f.write('alignas(8) const unsigned char model_quantized[] = {\n')
    f.write(',\n'.join(f'0x{b:02x}' for b in tflite_model))
    f.write('\n};\n\n')
    f.write(f'const int model_quantized_len = {len(tflite_model)};\n\n')
    f.write('#endif // MODEL_QUANTIZED_H\n')