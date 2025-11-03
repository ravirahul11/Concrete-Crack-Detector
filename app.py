import streamlit as st
import tensorflow as tf
# This import is critical for the custom_objects fix
from tensorflow.keras.applications.resnet50 import preprocess_input 
from PIL import Image
import numpy as np
import io
import os       # Import for file/path operations
import requests # Import for downloading the file

# --- Configuration ---

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!! PASTE YOUR GITHUB RELEASE DOWNLOAD LINK HERE !!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
MODEL_URL = "https://github.com/ravirahul11/Concrete-Crack-Detector/releases/download/v1.0/concrete_crack_detector_resnet50.h5" 

MODEL_PATH = "concrete_crack_detector_resnet50.h5"
IMG_SIZE = (224, 224)
CLASS_NAMES = ["Uncracked", "Cracked"]

# Set Streamlit page config
st.set_page_config(
    page_title="Concrete Crack Detector",
    page_icon="🧱",
    layout="centered",
)

# --- Model Loading ---
@st.cache_resource
def load_model():
    """
    Loads the trained Keras model.
    Downloads it first if it doesn't exist on the server.
    """
    # Check if model file already exists
    if not os.path.exists(MODEL_PATH):
        with st.spinner(f"Downloading model... (This is a one-time setup)"):
            try:
                # Download the file from the URL
                r = requests.get(MODEL_URL, allow_redirects=True)
                r.raise_for_status() # Check for download errors
                
                # Save the file
                with open(MODEL_PATH, 'wb') as f:
                    f.write(r.content)
                st.success("Model downloaded successfully!")
            except Exception as e:
                st.error(f"Error downloading model: {e}")
                return None

    # Now, load the model from the downloaded file
    try:
        # Define the custom object (the function) that Keras needs to find
        custom_objects = {"preprocess_input": preprocess_input}
        
        # Pass the custom_objects dictionary to load_model
        model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# --- Image Preprocessing ---
def preprocess_image(image_bytes):
    """
    Preprocesses the uploaded image to be model-ready.
    1. Opens the image
    2. Resizes it to IMG_SIZE
    3. Converts to NumPy array
    4. Expands dimensions to create a batch of 1
    """
    try:
        # Open image from bytes
        img = Image.open(io.BytesIO(image_bytes))
        
        # Resize to the model's expected input size
        img = img.resize(IMG_SIZE)
        
        # Convert to NumPy array
        img_array = np.array(img)
        
        # Ensure it has 3 channels (RGB)
        if img_array.ndim == 2: # Grayscale image
            img_array = np.stack((img_array,)*3, axis=-1)
        elif img_array.shape[2] == 4: # RGBA image
            img_array = img_array[:, :, :3]
            
        # Expand dimensions to create a batch (1, 224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0).astype('float32')
        
        return img_array
    
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

# --- Main App UI ---

st.title("🧱 Concrete Crack Detector")
st.markdown(
    "Upload an image of a concrete surface to classify it as **cracked** or **uncracked**."
)
st.markdown(
    "This app uses a **ResNet50** deep learning model trained on the "
    "[Concrete Crack Images for Classification dataset](https://www.kaggle.com/datasets/arunrk7/surface-crack-detection)."
)

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is None:
    st.info("Please upload an image to get started.")

elif model is None:
    st.error("Model could not be loaded. Please check logs.")

else:
    # --- Processing and Prediction ---
    
    # 1. Read and display the uploaded image
    image_bytes = uploaded_file.getvalue()
    st.image(image_bytes, caption="Uploaded Image", use_container_width=True) # Fixed parameter

    # 2. Preprocess the image
    processed_image = preprocess_image(image_bytes)
    
    if processed_image is not None:
        # 3. Make prediction
        with st.spinner("Analyzing the image..."):
            prediction = model.predict(processed_image)
        
        # The model outputs a single value (probability)
        # We check if it's > 0.5 to classify as "Cracked"
        score = prediction[0][0]
        if score > 0.5:
            pred_class = "Cracked"
            confidence = score * 100
            st.error(f"**Prediction: {pred_class}**")
        else:
            pred_class = "Uncracked"
            confidence = (1 - score) * 100
            st.success(f"**Prediction: {pred_class}**")
        
        # Display confidence
        st.metric(label="Confidence", value=f"{confidence:.2f}%")
        
        # Optional: Show a bar for confidence
        progress_bar = st.progress(0)
        progress_bar.progress(int(confidence), text=f"{confidence:.2f}%")

