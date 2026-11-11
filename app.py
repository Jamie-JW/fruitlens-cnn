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

# Fruit information shown after a prediction (EPIC #5: nutrition, health, ripeness, uses)
# Typical values for one medium fruit, not a measurement of the photo (source: USDA FoodData Central)
FRUIT_INFO = {
    "Apple": {
        "description": "A fruit with crisp flesh and a sweet or tart flavour.",
        "serving": "One medium apple (about 182 g)",
        "calories": "95 kcal",
        "carbohydrates": "25 g",
        "fibre": "4.4 g",
        "vitamin_c": "8.4 mg",
        "benefits": [
            "Provides dietary fibre, good for digestion",
            "Contains vitamin C and antioxidants",
        ],
        "ripeness": [
            "Ripe apples are firm, with smooth skin and no soft spots",
            "A sweet smell means it is ready to eat",
            "Wrinkled or soft skin means it is getting old",
        ],
        "uses": [
            "Eat fresh as a snack",
            "Bake into apple pie or crumble",
            "Slice into salads, or make apple juice",
        ],
        "storage": "Store in the refrigerator to help keep it fresh.",
    },
    "Banana": {
        "description": "A soft fruit with a naturally sweet flavour.",
        "serving": "One medium banana (about 118 g)",
        "calories": "105 kcal",
        "carbohydrates": "27 g",
        "fibre": "3.1 g",
        "vitamin_c": "10.3 mg",
        "benefits": [
            "Rich in potassium and vitamin B6",
            "Provides carbohydrates for quick energy",
        ],
        "ripeness": [
            "Green skin: not ripe yet",
            "Yellow skin: ripe and ready to eat",
            "Brown spots: very ripe and sweet, best for baking",
        ],
        "uses": [
            "Eat fresh as a quick snack",
            "Blend into a smoothie",
            "Bake banana bread or add to pancakes",
        ],
        "storage": "Keep at room temperature until ripe.",
    },
}


def show_fruit_info(fruit):
    """Show typical information about the predicted fruit in three tabs."""
    info = FRUIT_INFO.get(fruit)
    if info is None:  # e.g. an "Other" class has no information
        return
    st.subheader(f"About the {fruit.lower()}")
    st.write(info["description"])
    nutrition_tab, ripeness_tab, uses_tab = st.tabs(["🥗 Nutrition", "🕒 Ripeness", "🍰 Uses and storage"])
    with nutrition_tab:
        st.write(f"Typical values for: {info['serving']}")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Calories", info["calories"])
        col2.metric("Carbs", info["carbohydrates"])
        col3.metric("Fibre", info["fibre"])
        col4.metric("Vitamin C", info["vitamin_c"])
        st.markdown("\n".join(f"- {line}" for line in info["benefits"]))
        st.caption("Typical values (USDA FoodData Central). Real fruit varies with size and variety.")
    with ripeness_tab:
        st.markdown("\n".join(f"- {line}" for line in info["ripeness"]))
    with uses_tab:
        st.markdown("\n".join(f"- {line}" for line in info["uses"]))
        st.write(f"**Storage:** {info['storage']}")


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
        show_fruit_info(fruit)
