import streamlit as st
import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, ViTForImageClassification
from safetensors.torch import load_file


# ==========================================
# 1. Model Definition
# ==========================================
class BottleneckViT(nn.Module):
    def __init__(self, model_name, num_concepts=49, num_classes=1854):
        super(BottleneckViT, self).__init__()
        self.vit = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=num_concepts,
            ignore_mismatched_sizes=True
        )
        self.classifier = nn.Linear(num_concepts, num_classes)

    def forward(self, pixel_values):
        concepts = self.vit(pixel_values).logits
        logits = self.classifier(concepts)
        return logits, concepts


# ==========================================
# 2. Formatting Helpers for the UI
# ==========================================
def format_model_name(model_path):
    """
    Extracts a clean name from specific model filenames.
    Translates 'model_ind.safetensors' -> 'Independent'
    Translates 'model_joint_0,1.safetensors' -> 'Joint (Lambda = 0.1)'
    """
    filename = os.path.basename(model_path).replace(".safetensors", "")

    if filename.startswith("model_"):
        filename = filename.replace("model_", "", 1)

    if filename == "ind":
        return "Independent"
    elif filename == "seq":
        return "Sequential"
    elif filename.startswith("joint_"):
        lambda_val = filename.replace("joint_", "")
        lambda_val = lambda_val.replace(",", ".")
        return f"Joint (Lambda = {lambda_val})"

    return filename.replace('_', ' ').title()


def format_image_name(img_path):
    """Formats 'object_images/aardvark/aardvark_01.jpg' to 'Aardvark [aardvark_01.jpg]'"""
    norm_path = os.path.normpath(img_path)
    parts = norm_path.split(os.sep)
    if len(parts) >= 2:
        class_name = parts[-2].replace('_', ' ').title()
        file_name = parts[-1]
        return f"{class_name}  [{file_name}]"
    return os.path.basename(img_path)


def get_available_models(directory="models"):
    """Recursively finds all .safetensors files."""
    safetensors_files = []
    if os.path.exists(directory):
        for root, dirs, files in os.walk(directory):
            for f in files:
                if f.endswith(".safetensors"):
                    safetensors_files.append(os.path.join(root, f))
    return sorted(safetensors_files)


# ==========================================
# 3. Data & Environment Loading (Cached)
# ==========================================
@st.cache_resource
def load_base_environment():
    model_name = "google/vit-large-patch16-224"
    processor = AutoImageProcessor.from_pretrained(model_name)

    root_dir = "object_images"
    concepts_file = "spose_embedding_49d_sorted.txt"

    class_names = []
    demo_images = []
    concepts_matrix = None

    # Load ground truth concept matrix
    if os.path.exists(concepts_file):
        concepts_matrix = np.loadtxt(concepts_file)

    # Load classes and pick demo images
    if os.path.exists(root_dir):
        class_names = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])

        # Randomly select 50 classes for the dropdown demo to keep it fast
        np.random.seed(42)
        sample_size = min(50, len(class_names))
        if sample_size > 0:
            sample_classes = np.random.choice(class_names, sample_size, replace=False)
            sample_classes = sorted(sample_classes)

            for cls_name in sample_classes:
                cls_dir = os.path.join(root_dir, cls_name)
                imgs = [img for img in os.listdir(cls_dir) if img.lower().endswith(".jpg")]
                if imgs:
                    demo_images.append(os.path.join(cls_dir, imgs[0]))
    else:
        st.error("Error: 'object_images' folder not found. Please ensure it is in the same directory.")

    return processor, demo_images, class_names, concepts_matrix


@st.cache_resource
def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "google/vit-large-patch16-224"

    model = BottleneckViT(model_name, num_concepts=49, num_classes=1854)

    if os.path.exists(model_path):
        state_dict = load_file(model_path)
        model.load_state_dict(state_dict)
    else:
        st.error(f"Failed to load model from {model_path}")

    model.to(device)
    model.eval()
    return model, device


# ==========================================
# 4. Streamlit UI Construction
# ==========================================
st.set_page_config(layout="wide", page_title="Test-Time Intervention")
st.title("Test-Time Intervention - Concept Bottleneck Model")

# Check for available models
available_models = get_available_models("models")
if not available_models:
    st.error("No .safetensors models found in the 'models' directory.")
    st.stop()

# Sidebar Settings
with st.sidebar:
    st.header("⚙️ Settings")
    selected_model_path = st.selectbox(
        "Select Model",
        available_models,
        format_func=format_model_name
    )
    st.markdown("---")
    st.info("Adjust the concept sliders on the right to see how the model's final classification changes in real-time.")

# Load Base Data
processor, demo_images, class_names, concepts_matrix = load_base_environment()

if not demo_images:
    st.stop()

# Load the user-selected model
model, device = load_model(selected_model_path)

# Image Selection
selected_img_path = st.selectbox(
    "🖼️ Choose a test image",
    demo_images,
    format_func=format_image_name
)

if selected_img_path:
    img = Image.open(selected_img_path).convert("RGB")

    # Extract the true class from the folder structure (Ground Truth)
    true_class_name = os.path.basename(os.path.dirname(selected_img_path))
    if true_class_name in class_names:
        true_class_idx = class_names.index(true_class_name)
        true_concept_vector = concepts_matrix[true_class_idx] if concepts_matrix is not None else None
    else:
        true_class_idx = -1
        true_concept_vector = None

    # Extract the model's initial predicted concepts
    inputs = processor(images=img, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    with torch.no_grad():
        pred_concepts = model.vit(pixel_values).logits.squeeze(0).cpu().numpy()

    # State Management: Reset sliders if the user changes the image OR the model
    if ("current_img" not in st.session_state or
            st.session_state.current_img != selected_img_path or
            "current_model" not in st.session_state or
            st.session_state.current_model != selected_model_path):

        st.session_state.current_img = selected_img_path
        st.session_state.current_model = selected_model_path
        for i in range(49):
            st.session_state[f"c_{i}"] = float(pred_concepts[i])


    # Callbacks for Intervention Buttons
    def reset_predictions():
        for i in range(49):
            st.session_state[f"c_{i}"] = float(pred_concepts[i])


    def set_to_ground_truth():
        if true_concept_vector is not None:
            for i in range(49):
                st.session_state[f"c_{i}"] = float(true_concept_vector[i])


    # Build Layout
    col1, col2 = st.columns([1, 2.5])

    with col1:
        st.image(img, caption=f"True Class: {true_class_name.replace('_', ' ').title()}", use_container_width=True)

        st.markdown("### 🛠️ Intervention Controls")
        st.button("🔄 Reset to Model Predictions", on_click=reset_predictions, use_container_width=True)

        if true_concept_vector is not None:
            st.button("✨ Set All to Ground Truth", on_click=set_to_ground_truth, type="primary",
                      use_container_width=True)

    with col2:
        st.subheader("Concept Manipulation (49 Dimensions)")
        st.markdown("The **(GT)** value indicates the True Concept value from the dataset.")

        slider_cols = st.columns(7)
        updated_concepts = []

        # Generate the 49 Sliders
        for i in range(49):
            with slider_cols[i % 7]:
                gt_val = true_concept_vector[i] if true_concept_vector is not None else 0.0
                label = f"C{i}"
                if true_concept_vector is not None:
                    label += f" (GT:{gt_val:.2f})"

                # Dynamic slider limits to prevent out-of-bounds crashes
                min_v = float(min(-2.0, pred_concepts[i] - 1.5, gt_val - 1.0))
                max_v = float(max(4.0, pred_concepts[i] + 1.5, gt_val + 1.0))

                # Failsafe bounds check for session state
                current_val = st.session_state[f"c_{i}"]
                if current_val < min_v: st.session_state[f"c_{i}"] = min_v
                if current_val > max_v: st.session_state[f"c_{i}"] = max_v

                val = st.slider(
                    label,
                    min_value=min_v,
                    max_value=max_v,
                    step=0.01,
                    key=f"c_{i}"  # Binding strictly via key, no 'value=' arg
                )
                updated_concepts.append(val)

    # Forward pass the manipulated concepts through the final classifier layer
    updated_concepts_tensor = torch.tensor([updated_concepts], dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model.classifier(updated_concepts_tensor).squeeze(0)
        probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()

    top5_indices = np.argsort(probs)[-5:][::-1]

    st.divider()
    st.subheader("🏆 Top 5 Predictions (Final Classification)")

    metrics_cols = st.columns(5)
    for i, idx in enumerate(top5_indices):
        with metrics_cols[i]:
            is_correct = (idx == true_class_idx)
            label_color = "🟢" if is_correct else "🔴"
            display_name = class_names[idx].replace('_', ' ').title()
            st.metric(label=f"{label_color} {display_name}", value=f"{probs[idx] * 100:.2f}%")