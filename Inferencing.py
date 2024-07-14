import onnxruntime as ort
import numpy as np
from PIL import Image

# Load the ONNX model
onnx_path = "model.onnx"
ort_session = ort.InferenceSession(onnx_path)

# Define a function to preprocess the input image
def preprocess(image_path):
    img = Image.open(image_path).resize((150, 150))
    img = np.array(img).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img

# Perform inference
input_image = preprocess('data/test/water_bottle/test_image.jpg')
inputs = {ort_session.get_inputs()[0].name: input_image}
outputs = ort_session.run(None, inputs)
print(f"Model output: {outputs}")
