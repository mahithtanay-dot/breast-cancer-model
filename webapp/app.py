"""
Gradio demo for the Breast Cancer Classifier project.
Deploy this on a free Hugging Face Space (Settings -> SDK: Gradio).

Before deploying, train both models in Colab, push them to your own Hugging
Face Hub repos using the "Publish to Hugging Face Hub" cells at the end of
each notebook, then update TABULAR_ADAPTER_REPO and IMAGE_MODEL_REPO below.
"""

import gradio as gr
import torch
import torch.nn as nn
import torchvision.transforms as T
from huggingface_hub import hf_hub_download
from PIL import Image

# --- Change these to your own Hugging Face Hub repo IDs after training ---
TABULAR_ADAPTER_REPO = "YOUR_HF_USERNAME/breast-cancer-tabular-lora"
IMAGE_MODEL_REPO = "YOUR_HF_USERNAME/breast-cancer-histopathology-cnn"
BASE_MODEL_NAME = "Qwen/Qwen2.5-3B"

TARGET_NAMES = ["benign", "malignant"]
FEATURE_BASE_NAMES = [
    "radius", "texture", "perimeter", "area", "smoothness",
    "compactness", "concavity", "concave_points", "symmetry", "fractal_dimension",
]
FEATURE_ORDER = [f"{stat}_{name}" for stat in ["mean", "se", "worst"] for name in FEATURE_BASE_NAMES]

# Reasonable defaults so the sliders start on a plausible-looking sample
DEFAULT_VALUES = {
    "mean_radius": 14.0, "mean_texture": 20.0, "mean_perimeter": 90.0, "mean_area": 600.0,
    "mean_smoothness": 0.1, "mean_compactness": 0.1, "mean_concavity": 0.1, "mean_concave_points": 0.05,
    "mean_symmetry": 0.18, "mean_fractal_dimension": 0.06,
    "se_radius": 0.4, "se_texture": 1.0, "se_perimeter": 3.0, "se_area": 40.0,
    "se_smoothness": 0.007, "se_compactness": 0.02, "se_concavity": 0.03, "se_concave_points": 0.01,
    "se_symmetry": 0.02, "se_fractal_dimension": 0.003,
    "worst_radius": 16.0, "worst_texture": 25.0, "worst_perimeter": 105.0, "worst_area": 800.0,
    "worst_smoothness": 0.14, "worst_compactness": 0.25, "worst_concavity": 0.27, "worst_concave_points": 0.11,
    "worst_symmetry": 0.29, "worst_fractal_dimension": 0.08,
}

DISCLAIMER = (
    "**For fun and education only — not a medical device, not for real diagnosis.** "
    "These models are trained on small public research datasets and have not been "
    "clinically validated. See the [GitHub repo](https://github.com/mahithtanay-dot/breast-cancer-model) "
    "for the full model card and limitations."
)

# ---------------------------------------------------------------------------
# Image model (small CNN) — fast on free CPU hardware
# ---------------------------------------------------------------------------


class PatchCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, 2))

    def forward(self, x):
        return self.classifier(self.features(x))


_image_model = None
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
eval_transform = T.Compose([T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def load_image_model():
    global _image_model
    if _image_model is None:
        path = hf_hub_download(IMAGE_MODEL_REPO, "best_model.pt")
        model = PatchCNN()
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        _image_model = model
    return _image_model


def predict_image(img):
    if img is None:
        return None
    model = load_image_model()
    x = eval_transform(img.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    return {"non-cancerous": float(probs[0]), "cancerous": float(probs[1])}


# ---------------------------------------------------------------------------
# Tabular model (Qwen2.5-3B + LoRA) — loads in full precision on CPU since
# bitsandbytes 4-bit quantization requires a GPU. Slow (roughly 30-60s per
# prediction on free CPU hardware) — that's expected, not a bug.
# ---------------------------------------------------------------------------

_tabular_model = None
_tabular_tokenizer = None


def load_tabular_model():
    global _tabular_model, _tabular_tokenizer
    if _tabular_model is None:
        from peft import PeftModel
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL_NAME, num_labels=2, torch_dtype=torch.float32
        )
        base_model.config.pad_token_id = tokenizer.pad_token_id

        model = PeftModel.from_pretrained(base_model, TABULAR_ADAPTER_REPO)
        model.eval()

        _tabular_model, _tabular_tokenizer = model, tokenizer
    return _tabular_model, _tabular_tokenizer


def predict_tabular(*values):
    model, tokenizer = load_tabular_model()
    measurements = dict(zip(FEATURE_ORDER, values))
    text = "; ".join(f"{col}: {measurements[col]:.4f}" for col in FEATURE_ORDER)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    return {TARGET_NAMES[0]: float(probs[0]), TARGET_NAMES[1]: float(probs[1])}


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="Breast Cancer Classifier Demo") as demo:
    gr.Markdown("# Breast Cancer Classifier Demo")
    gr.Markdown(DISCLAIMER)

    with gr.Tab("Tissue image (fast)"):
        gr.Markdown(
            "Upload a 50x50 pixel breast tissue histopathology patch. "
            "Sample patches are available in the [IDC dataset](https://andrewjanowczyk.com/use-case-6-invasive-ductal-carcinoma-idc-segmentation/)."
        )
        img_input = gr.Image(type="pil", label="Tissue patch")
        img_button = gr.Button("Classify", variant="primary")
        img_output = gr.Label(label="Prediction")
        img_button.click(predict_image, inputs=img_input, outputs=img_output)

    with gr.Tab("Numeric measurements (slow — runs a 3B parameter model on CPU)"):
        gr.Markdown(
            "Enter cytology measurements from a cell sample. This tab runs a "
            "3-billion-parameter language model on free CPU hardware, so each "
            "prediction can take 30-60+ seconds. That's expected."
        )
        number_inputs = [
            gr.Number(label=feature, value=DEFAULT_VALUES[feature]) for feature in FEATURE_ORDER
        ]
        tab_button = gr.Button("Classify", variant="primary")
        tab_output = gr.Label(label="Prediction")
        tab_button.click(predict_tabular, inputs=number_inputs, outputs=tab_output)

if __name__ == "__main__":
    demo.launch()
