"""
Predict pneumonia + stage from a single DICOM X-ray.

Usage:
    python -m scripts.predict --image data/raw/stage_2_train_images/0004cfab-....dcm
    python -m scripts.predict --image path/to/xray.dcm --gradcam

Output:
    Prediction: NORMAL or PNEUMONIA
    Confidence: XX%
    Stage:      MILD / MODERATE / SEVERE (only if pneumonia)
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pydicom
import torch
from PIL import Image

from src.data.transforms import get_eval_transforms
from src.models.classifier import PneumoniaClassifier

STAGE_NAMES = {0: "MILD", 1: "MODERATE", 2: "SEVERE"}
LABEL_NAMES = {0: "NORMAL", 1: "PNEUMONIA"}


def load_dicom_as_pil(path: str) -> Image.Image:
    """Load a DICOM X-ray and convert to RGB PIL image."""
    ds = pydicom.dcmread(path)
    pixels = ds.pixel_array.astype(np.float32)

    # Normalize to 0-255
    pixels = pixels - pixels.min()
    if pixels.max() > 0:
        pixels = pixels / pixels.max() * 255.0
    pixels_uint8 = pixels.astype(np.uint8)

    # Grayscale → RGB
    rgb = cv2.cvtColor(pixels_uint8, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(rgb)


def predict(image_path: str, model, device):
    """Run inference on a single X-ray."""
    # Load + transform
    img = load_dicom_as_pil(image_path)
    transform = get_eval_transforms(224)
    tensor = transform(img).unsqueeze(0).to(device)

    # Predict
    model.eval()
    with torch.no_grad():
        binary_logits, stage_logits = model(tensor)
        binary_probs = torch.softmax(binary_logits, dim=1)[0].cpu().numpy()
        stage_probs = torch.softmax(stage_logits, dim=1)[0].cpu().numpy()

    binary_pred = int(np.argmax(binary_probs))
    stage_pred = int(np.argmax(stage_probs))

    return {
        "label": LABEL_NAMES[binary_pred],
        "confidence": float(binary_probs[binary_pred]),
        "probs": {
            "NORMAL": float(binary_probs[0]),
            "PNEUMONIA": float(binary_probs[1]),
        },
        "stage": STAGE_NAMES[stage_pred] if binary_pred == 1 else "N/A",
        "stage_confidence": float(stage_probs[stage_pred]) if binary_pred == 1 else 0.0,
        "stage_probs": {
            "MILD": float(stage_probs[0]),
            "MODERATE": float(stage_probs[1]),
            "SEVERE": float(stage_probs[2]),
        },
    }


def pretty_print(image_path: str, result: dict):
    """Nicely formatted output."""
    print("")
    print("=" * 60)
    print(f"Image: {Path(image_path).name}")
    print("=" * 60)
    print(f"Prediction: {result['label']}")
    print(f"Confidence: {result['confidence'] * 100:.2f}%")
    print("")
    print("Probability breakdown:")
    print(f"  NORMAL    : {result['probs']['NORMAL'] * 100:.2f}%")
    print(f"  PNEUMONIA : {result['probs']['PNEUMONIA'] * 100:.2f}%")

    if result["label"] == "PNEUMONIA":
        print("")
        print(f"Stage: {result['stage']}")
        print(f"Stage confidence: {result['stage_confidence'] * 100:.2f}%")
        print("")
        print("Stage breakdown:")
        print(f"  MILD      : {result['stage_probs']['MILD'] * 100:.2f}%")
        print(f"  MODERATE  : {result['stage_probs']['MODERATE'] * 100:.2f}%")
        print(f"  SEVERE    : {result['stage_probs']['SEVERE'] * 100:.2f}%")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Predict pneumonia from a chest X-ray")
    parser.add_argument("--image", type=str, required=True, help="Path to .dcm file")
    parser.add_argument("--checkpoint", type=str, default="models/best.pt")
    args = parser.parse_args()

    if not Path(args.image).exists():
        print(f"ERROR: Image not found: {args.image}")
        return

    if not Path(args.checkpoint).exists():
        print(f"ERROR: Model not found: {args.checkpoint}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print("Loading model...")
    model = PneumoniaClassifier(multi_task=True).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"Model loaded (trained epoch {ckpt.get('epoch', '?')}, val AUC {ckpt.get('best_auc', 0):.4f})")

    # Predict
    result = predict(args.image, model, device)
    pretty_print(args.image, result)


if __name__ == "__main__":
    main()