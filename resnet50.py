import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pathlib
import urllib.request
import os

# Model path
MODEL_PATH = 'inception_model.keras'
MODEL_URL = 'https://github.com/DARK-art108/Cotton-Leaf-Disease-Prediction/releases/download/v1.0/model_resnet.hdf5'

# Download model if not present
if not pathlib.Path(MODEL_PATH).is_file():
    st.write(f'Model {MODEL_PATH} not found. Downloading...')
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# Load your trained model
model = load_model(MODEL_PATH)

def model_predict(img, model):
    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 150, 150, 3), dtype=np.float32)
    
    # Image sizing
    size = (150, 150)
    image = ImageOps.fit(img, size, Image.LANCZOS)
    
    # Turn the image into a numpy array
    image_array = np.asarray(image)
    
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    
    # Load the image into the array
    data[0] = normalized_image_array
    
    # Print the shape of the preprocessed image data
    st.write("Shape of preprocessed image data:", data.shape)
    
    # Run the inference
    prediction = model.predict(data)
    
    # Print raw prediction
    st.write("Raw prediction:", prediction)
    
    return np.argmax(prediction, axis=1)  # return position of the highest probability

# Streamlit interface
st.title("Cotton Leaf Disease Prediction")
st.header("Transfer Learning Using Inception")
st.text("Upload a Cotton Leaf Disease or Non-Diseased Image")

uploaded_file = st.file_uploader("Choose a Cotton Leaf Image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Cotton Leaf Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label = model_predict(image, model)
    st.write(f"Predicted label: {label}")
    if label == 0:
        st.write("The leaf is a diseased cotton leaf.")
    elif label == 1:
        st.write("The leaf is a diseased cotton plant.")
    elif label == 2:
        st.write("The leaf is a fresh cotton leaf.")
    else:
        st.write("The leaf is a fresh cotton plant.")
