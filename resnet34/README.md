# FruitLens: ResNet34 version (alternative model)

This folder is a **separate** version of FruitLens that uses a PyTorch **ResNet34** model.
It does not change the main app (`app.py` and `models/` in the repository root).

## Model
- File: `models/best_resnet34_fruit.pth` (85 MB, PyTorch weights)
- Architecture: torchvision ResNet34, last layer replaced with 3 outputs: **Apple, Banana, Other**
- Input: resize to 224 × 224, pixels 0–1, ImageNet normalisation
- Confidence threshold: 90% (below this the app shows "Unknown")
- Reported test accuracy: 98.99%

**Source:** this model was trained by a FruitLens team member as part of **another course**, and is included here as a comparison / backup model, not as the team's main CNN.

## Run locally
```
pip install -r resnet34/requirements.txt
streamlit run resnet34/app_resnet34.py
```

## Deploy on Streamlit Community Cloud
Create a new app from the same repository and branch, and set **Main file path** to `resnet34/app_resnet34.py`.
Streamlit uses `resnet34/requirements.txt` (the CPU-only PyTorch build) for this app.
