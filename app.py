from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps
import keras

# Model settings (confirmed by testing with real apple and banana photos)
MODEL_PATH = Path(__file__).parent / "models" / "model_a_custom_cnn.keras"
IMAGE_SIZE = (224, 224)
CLASS_NAMES = ["Apple", "Banana"]  # output 0 = apple, output 1 = banana

# Configure the browser page
st.set_page_config(
    page_title="FruitLens",
    page_icon="🍎"
)

# Increase font sizes
st.markdown(
    """
    <style>

    [data-testid="stMainBlockContainer"] {
    padding-top: 3rem !important;
}
    p {
        font-size: 18px !important;
    }

    [data-testid="stFileUploader"] label {
        font-size: 18px !important;
    }

    .stButton button {
        font-size: 16px !important;
        padding: 8px 18px;
    }

    [data-testid="stAlert"] p {
        font-size: 16px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the CNN model once and reuse it (Streamlit re-runs this file on every click)
@st.cache_resource
def load_model():
    return keras.models.load_model(MODEL_PATH)


def predict(image):
    """Return (fruit name, confidence) for a PIL image."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = image.resize(IMAGE_SIZE, Image.BILINEAR)
    pixels = np.asarray(image, dtype="float32")  # keep 0-255, no division
    probabilities = load_model().predict(pixels[None], verbose=0)[0]
    best = int(np.argmax(probabilities))
    return CLASS_NAMES[best], float(probabilities[best])


# Application heading
st.title("🍎 FruitLens 🍌")
st.write("Upload a photo of an apple or banana.")

# Allow the user to upload an image
uploaded_file = st.file_uploader(
    "Choose a fruit image",
    type=["jpg", "jpeg", "png"]
)

# Continue only after an image has been uploaded
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # Display the uploaded image
    st.image(
        image,
        caption="Uploaded image",
        width=180
    )

        # Classify the uploaded image with the CNN model
    if st.button("Classify Fruit"):
        with st.spinner("Recognising fruit..."):
            fruit, confidence = predict(image)
        st.success(f"Prediction: {fruit} ({confidence:.0%} confidence)")
