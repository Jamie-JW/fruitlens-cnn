"""FruitLens: alternative version using a PyTorch ResNet34 model.

Kept in its own folder so it does not mix with the main app (app.py in the repo root).
Run locally:  streamlit run resnet34/app_resnet34.py
"""
import os
import streamlit as st
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from PIL import Image, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__))
CLASSES = ["Apple", "Banana", "Other"]
THRESHOLD = 0.90  # chosen by the model author from validation results


@st.cache_resource
def load_model():
    """ResNet34 with a 3-class last layer (Apple, Banana, Other)."""
    model = torchvision.models.resnet34()
    model.fc = nn.Linear(512, 3)
    state = torch.load(os.path.join(HERE, "models", "best_resnet34_fruit.pth"), map_location="cpu")
    model.load_state_dict(state)
    return model.eval()


# Resize to 224x224, pixels 0-1, ImageNet normalisation (no random augmentation)
PREPROCESS = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

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


st.set_page_config(page_title="FruitLens (ResNet34)", page_icon="🔍")
st.title("🔍 FruitLens: ResNet34 version")
st.caption("Alternative model (PyTorch ResNet34). The main FruitLens app is separate.")

model = load_model()
uploaded = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image = ImageOps.exif_transpose(Image.open(uploaded)).convert("RGB")
    st.image(image, width=220)

    if st.button("Classify Fruit"):
        x = PREPROCESS(image).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)[0]
        i = int(probs.argmax())
        conf = float(probs[i])
        threshold = THRESHOLD

        if conf < threshold:
            st.warning(f"Unknown: the model is not sure ({conf:.0%}, needs {threshold:.0%}).")
        elif CLASSES[i] == "Other":
            st.error(f"Not an apple or banana ({conf:.0%})")
        else:
            st.success(f"{CLASSES[i]} ({conf:.0%})")
            show_fruit_info(CLASSES[i])

        with st.expander("All outputs"):
            for name, p in zip(CLASSES, probs):
                st.progress(float(p), text=f"{name}: {float(p):.1%}")
