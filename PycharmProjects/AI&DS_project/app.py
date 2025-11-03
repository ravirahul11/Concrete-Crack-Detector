import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import io
from tensorflow.keras.applications.resnet50 import preprocess_input

# --- Configuration ---

MODEL_PATH = "concrete_crack_detector_resnet50.h5"
IMG_SIZE = (224, 224)
CLASS_NAMES = ["Uncracked", "Cracked"]

# Configure Streamlit app layout and metadata
st.set_page_config(
    page_title="Concrete Crack Detector",
    page_icon="🧱",
    layout="centered",
)


# --- Model Loading ---
@st.cache_resource
def load_model():
    """
    Loads the trained model once and caches it for faster performance.
    """
    try:
        # Some models may use custom functions (like preprocess_input)
        custom_objects = {"preprocess_input": preprocess_input}

        model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


model = load_model()


# --- Image Preprocessing ---
def preprocess_image(image_bytes):
    """
    Converts the uploaded image into a format suitable for prediction.
    Steps:
    - Opens the image from memory
    - Resizes it to match the model input
    - Converts it to an array and ensures 3 channels (RGB)
    - Expands dimensions to create a batch of one image
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = img.resize(IMG_SIZE)
        img_array = np.array(img)

        # Handle grayscale or images with alpha channel
        if img_array.ndim == 2:
            img_array = np.stack((img_array,) * 3, axis=-1)
        elif img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]

        img_array = np.expand_dims(img_array, axis=0).astype('float32')
        return img_array

    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None


# --- Main App UI ---

st.title("🧱 Concrete Crack Detector")
st.markdown(
    "Upload an image of a concrete surface to check whether it is **cracked** or **uncracked**."
)
st.markdown(
    "This tool uses a **ResNet50** model trained on the "
    "[Concrete Crack Images for Classification dataset](https://www.kaggle.com/datasets/arunrk7/surface-crack-detection)."
)

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is None:
    st.info("Please upload an image to get started.")

elif model is None:
    st.error("Model could not be loaded. Please check the model file.")

else:
    # --- Processing and Prediction ---

    # Display the uploaded image
    image_bytes = uploaded_file.getvalue()
    st.image(image_bytes, caption="Uploaded Image", use_container_width=True)

    # Preprocess image for model input
    processed_image = preprocess_image(image_bytes)

    if processed_image is not None:
        # Run model prediction
        with st.spinner("Analyzing the image..."):
            prediction = model.predict(processed_image)

        # Output interpretation
        score = prediction[0][0]
        if score > 0.5:
            pred_class = "Cracked"
            confidence = score * 100
            st.error(f"**Prediction: {pred_class}**")
        else:
            pred_class = "Uncracked"
            confidence = (1 - score) * 100
            st.success(f"**Prediction: {pred_class}**")

        # Display prediction confidence
        st.metric(label="Confidence", value=f"{confidence:.2f}%")

        # Visual confidence indicator
        progress_bar = st.progress(0)
        progress_bar.progress(int(confidence), text=f"{confidence:.2f}%")
