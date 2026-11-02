
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import gdown

from pathlib import Path
from PIL import Image
from torchvision import transforms
from torchvision.models import densenet121

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Chest X-ray AI",
    page_icon="🩻",
    layout="wide"
)

st.title("🩻 Chest X-ray Classification")
st.write("DenseNet121 with Grad-CAM Explainability")

st.warning(
    "Educational project only. This AI tool is not a medical "
    "diagnostic system. Consult a qualified healthcare professional."
)

# -----------------------------
# Device and model configuration
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "densenet121_chest_xray_finetuned.pth"

GOOGLE_DRIVE_FILE_ID = "1D39y4yAUYJ1COTmBiSJ7RcvBLucKg_ED"

# -----------------------------
# Download model automatically
# -----------------------------
def download_model():

    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 1000000:
        return

    st.info("Downloading trained model from Google Drive...")

    try:
        result = gdown.download(
            id=GOOGLE_DRIVE_FILE_ID,
            output=str(MODEL_PATH),
            quiet=False
        )

        if result is None or not MODEL_PATH.exists():
            raise RuntimeError("Model download failed.")

        if MODEL_PATH.stat().st_size < 1000000:
            MODEL_PATH.unlink(missing_ok=True)
            raise RuntimeError("Downloaded model file is incomplete.")

        st.success("Model downloaded successfully!")

    except Exception as error:
        raise RuntimeError(
            f"Could not download model from Google Drive: {error}"
        )


# -----------------------------
# Load trained model
# -----------------------------
@st.cache_resource
def load_model():

    download_model()

    model = densenet121(weights=None)
    model.classifier = nn.Linear(
        model.classifier.in_features, 2
    )

    state_dict = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


# -----------------------------
# Image preprocessing
# -----------------------------
eval_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -----------------------------
# Grad-CAM Explainability
# -----------------------------
def generate_gradcam(model, image_tensor, class_index):

    activations = []
    gradients = []

    target_layer = model.features.denseblock4

    def forward_hook(module, inputs, output):
        activations.append(output)
        output.register_hook(
            lambda grad: gradients.append(grad)
        )

    handle = target_layer.register_forward_hook(forward_hook)

    try:
        model.zero_grad(set_to_none=True)

        with torch.enable_grad():
            input_tensor = image_tensor.detach().requires_grad_(True)
            output = model(input_tensor)
            output[0, class_index].backward()

    finally:
        handle.remove()

    if not activations or not gradients:
        raise RuntimeError("Grad-CAM gradients not found.")

    activation = activations[0]
    gradient = gradients[0]

    weights = gradient.mean(dim=(2, 3), keepdim=True)

    cam = (weights * activation).sum(dim=1, keepdim=True)
    cam = F.relu(cam)

    cam = F.interpolate(
        cam,
        size=(224, 224),
        mode="bilinear",
        align_corners=False
    )

    cam = cam[0, 0].detach().cpu().numpy()

    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    threshold = np.percentile(cam, 60)
    cam = np.maximum(cam - threshold, 0)

    cam = cam / (cam.max() + 1e-8)
    cam = cam ** 0.7

    return cam


# -----------------------------
# Upload X-ray image
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload a single chest X-ray image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    try:
        original_image = Image.open(uploaded_file).convert("RGB")

        image_tensor = eval_transform(
            original_image
        ).unsqueeze(0).to(device)

        model = load_model()

        # Prediction
        with torch.no_grad():
            outputs = model(image_tensor)

            probabilities = torch.softmax(outputs, dim=1)[0]

            predicted_class = outputs.argmax(dim=1).item()

            confidence = probabilities[predicted_class].item()

        class_names = ["NORMAL", "PNEUMONIA"]

        # Grad-CAM
        cam = generate_gradcam(
            model,
            image_tensor,
            predicted_class
        )

        original_array = np.array(
            original_image.resize((224, 224))
        )

        heatmap_rgb = (
            plt.get_cmap("jet")(cam)[:, :, :3] * 255
        ).astype(np.uint8)

        alpha = (cam * 0.55)[..., np.newaxis]

        overlay = (
            original_array.astype(float) * (1 - alpha)
            + heatmap_rgb.astype(float) * alpha
        ).clip(0, 255).astype(np.uint8)

        # -----------------------------
        # Prediction results
        # -----------------------------
        st.subheader("Prediction Result")

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                original_image,
                caption="Uploaded X-ray",
                use_container_width=True
            )

        with col2:
            st.metric(
                "Predicted class",
                class_names[predicted_class]
            )

            st.metric(
                "Model confidence",
                f"{confidence:.2%}"
            )

            st.write("Class probabilities:")
            st.write(f"NORMAL: {probabilities[0].item():.2%}")
            st.write(f"PNEUMONIA: {probabilities[1].item():.2%}")

        # -----------------------------
        # Grad-CAM results
        # -----------------------------
        st.subheader("Grad-CAM Explainability")

        cam_col1, cam_col2, cam_col3 = st.columns(3)

        with cam_col1:
            st.image(
                original_array,
                caption="Original X-ray",
                use_container_width=True
            )

        with cam_col2:
            st.image(
                heatmap_rgb,
                caption="Influential regions heatmap",
                use_container_width=True
            )

        with cam_col3:
            st.image(
                overlay,
                caption="Heatmap overlay",
                use_container_width=True
            )

        st.caption(
            "Red/yellow regions indicate areas that relatively influenced "
            "the model prediction. They do not confirm the presence, exact "
            "location, or severity of infection."
        )

    except Exception as error:
        st.error(f"Could not process the image: {error}")
