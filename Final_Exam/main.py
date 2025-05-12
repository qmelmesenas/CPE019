import streamlit as st
import gdown
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

# Download model if not already present
model_path = "final_garbage_model.h5"
if not os.path.exists(model_path):
    gdown.download("https://drive.google.com/uc?id=1kIo9dns6-enGso6fHaIJEYDjJcQ3YBCg", model_path, quiet=False, fuzzy=True)

# Load the model
model = load_model(model_path)

# Class names
class_names = ['battery','biological','brown-glass','cardboard','clothes','green-glass','metal','paper','plastic','shoes','trash','white-glass']  

st.title("Garbage Classifier")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_container_width=True)

    # Preprocess image
    img = img.resize((150, 150))  
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    st.write(f"Predicted class: {class_names[class_index]}")
