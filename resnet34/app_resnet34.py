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

        st.write("All outputs:")
        for name, p in zip(CLASSES, probs):
            st.progress(float(p), text=f"{name}: {float(p):.1%}")
