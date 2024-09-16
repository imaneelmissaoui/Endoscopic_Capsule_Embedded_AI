import streamlit as st
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import joblib
import tensorflow as tf
import os

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="../Model/mobilenetv2_feature_extractor.tflite")
interpreter.allocate_tensors()

# Load the trained SVM model
svm_model = joblib.load("../Model/svm_model.pkl")

# Define class names
class_names = ['Normal', 'Ulcer', 'Polyps', 'Esophagitis']

# Function to extract features using the TensorFlow Lite model
def extract_features(image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Prepare the input data
    input_data = np.array(image, dtype=np.float32)
    input_data = np.expand_dims(input_data, axis=0)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Extract the features
    features = interpreter.get_tensor(output_details[0]['index'])
    return features

# Function to extract the real class from the filename
def get_real_class_from_filename(filename):
    # Remove the file extension
    filename = os.path.splitext(filename)[0]
    
    # Split the filename by underscores
    parts = filename.split('_')
    
    # The second part should be the class name
    if len(parts) >= 2:
        class_label = parts[1].lower()  # Convert class label to lowercase for consistency
        return class_label
    else:
        raise ValueError("Filename does not contain enough parts to extract the class.")

# Streamlit app
st.title("Medical Image Classification")
st.write("Upload an endoscopic image and classify it using the Hybrid MobileNetV2-SVM model.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Preprocess the image to the right format
    image = image.resize((224, 224))
    image_array = img_to_array(image) / 255.0  # Normalize

    # Extract features using the TensorFlow Lite model
    features = extract_features(image_array)

    # Flatten features if needed (depends on the model's output shape)
    features = np.squeeze(features)

    # Predict using the SVM model
    prediction = svm_model.predict(features.reshape(1, -1))

    # Get the predicted class
    predicted_class = class_names[int(prediction[0])].lower()  # Convert to lowercase

    # Extract the real class from the filename
    real_class = get_real_class_from_filename(uploaded_file.name)

    # Display both predicted and real class
    st.write(f"**Predicted Class:** {predicted_class.capitalize()}")  # Capitalize first letter
    st.write(f"**Real Class:** {real_class.capitalize()}")  # Capitalize first letter

    # Optionally, show whether the prediction was correct
    if predicted_class == real_class:
        st.success("The model predicted correctly!")
    else:
        st.error("The model's prediction was incorrect.")
