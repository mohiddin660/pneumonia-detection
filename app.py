"""
Pneumonia Detection Web UI (Accepts DCM, PNG, JPG with Smart Validation).

Run:  python app.py
Then open: http://localhost:5000
"""
import base64
import io
import time
from pathlib import Path

import cv2
import numpy as np
import pydicom
import torch
import torch.nn.functional as F
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from PIL import Image

from src.data.transforms import get_eval_transforms
from src.models.classifier import PneumoniaClassifier

# ---------------- CONFIG ----------------
app = Flask(__name__)
app.secret_key = "super_secret_medical_key_123"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload

UPLOAD_DIR = Path("static/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

STAGE_NAMES = {0: "MILD", 1: "MODERATE", 2: "SEVERE"}
LABEL_NAMES = {0: "NORMAL", 1: "PNEUMONIA"}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = None

# ---- OPTIMAL THRESHOLD ----
OPTIMAL_THRESHOLD = 0.67

# ---------------- MODEL LOADING ----------------
def load_model():
    global MODEL
    if MODEL is not None:
        return MODEL

    print("Loading model...")
    model = PneumoniaClassifier(multi_task=True).to(DEVICE)
    ckpt = torch.load("models/best.pt", map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    MODEL = model
    print(f"Model loaded on {DEVICE}")
    return model

# ---------------- IMAGE LOADING ----------------
def load_image(file_bytes: bytes, filename: str) -> np.ndarray:
    if filename.lower().endswith(".dcm"):
        ds = pydicom.dcmread(io.BytesIO(file_bytes))
        pixels = ds.pixel_array.astype(np.float32)
        pixels = pixels - pixels.min()
        if pixels.max() > 0:
            pixels = pixels / pixels.max() * 255.0
        img = pixels.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        pil = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        img = np.array(pil)
    return img

# ---------------- SMART VALIDATOR (For JPG/PNG) ----------------
def validate_image(img_rgb: np.ndarray, filename: str) -> tuple[bool, str]:
    """
    If it's a DICOM, trust it.
    If it's a JPG/PNG, check if it looks like a medical X-ray to block trees/anime.
    """
    if filename.lower().endswith(".dcm"):
        return True, ""

    # For JPG/PNG: Check size
    h, w = img_rgb.shape[:2]
    if h < 150 or w < 150:
        return False, "Image is too small. Please upload a standard chest X-ray."

    # For JPG/PNG: Check if grayscale (Block colorful images like anime)
    r, g, b = img_rgb[:,:,0], img_rgb[:,:,1], img_rgb[:,:,2]
    color_diff = np.mean(np.abs(r.astype(int) - g.astype(int))) + np.mean(np.abs(g.astype(int) - b.astype(int)))
    if color_diff > 15:
        return False, "Invalid Image: Please upload a grayscale Chest X-ray. Colorful images are rejected."

    # For JPG/PNG: Check brightness (Block black/white trees)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    mean_brightness = np.mean(gray)
    if mean_brightness > 230 or mean_brightness < 15:
        return False, "Invalid Image: Image brightness does not match a medical X-ray."

    return True, ""

# ---------------- GRAD-CAM ----------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=1):
        self.model.zero_grad()
        output = self.model(input_tensor)
        logits = output[0] if isinstance(output, tuple) else output
        score = logits[0, class_idx]
        score.backward()
        grads = self.gradients[0].cpu().numpy()
        acts = self.activations[0].cpu().numpy()
        weights = grads.mean(axis=(1, 2))
        cam = np.zeros(acts.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * acts[i]
        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam

def make_gradcam_overlay(model, img_rgb: np.ndarray, class_idx: int = 1) -> np.ndarray:
    transform = get_eval_transforms(224)
    tensor = transform(Image.fromarray(img_rgb)).unsqueeze(0).to(DEVICE)
    target_layer = model.backbone.conv_head
    gradcam = GradCAM(model, target_layer)
    cam = gradcam.generate(tensor, class_idx=class_idx)
    cam_resized = cv2.resize(cam, (img_rgb.shape[1], img_rgb.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (0.6 * img_rgb + 0.4 * heatmap).astype(np.uint8)
    return overlay

# ---------------- PREDICTION ----------------
def predict(img_rgb: np.ndarray) -> dict:
    model = load_model()
    transform = get_eval_transforms(224)
    tensor = transform(Image.fromarray(img_rgb)).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        binary_logits, stage_logits = model(tensor)
        binary_probs = F.softmax(binary_logits, dim=1)[0].cpu().numpy()
        stage_probs = F.softmax(stage_logits, dim=1)[0].cpu().numpy()

    binary_pred = 1 if binary_probs[1] >= OPTIMAL_THRESHOLD else 0
    stage_pred = int(np.argmax(stage_probs))
    overlay = make_gradcam_overlay(model, img_rgb, class_idx=binary_pred)

    return {
        "label": LABEL_NAMES[binary_pred],
        "confidence": float(binary_probs[binary_pred]) * 100,
        "normal_prob": float(binary_probs[0]) * 100,
        "pneumonia_prob": float(binary_probs[1]) * 100,
        "stage": STAGE_NAMES[stage_pred] if binary_pred == 1 else None,
        "stage_confidence": float(stage_probs[stage_pred]) * 100 if binary_pred == 1 else 0,
        "mild_prob": float(stage_probs[0]) * 100,
        "moderate_prob": float(stage_probs[1]) * 100,
        "severe_prob": float(stage_probs[2]) * 100,
        "gradcam_overlay": overlay,
        "threshold_used": OPTIMAL_THRESHOLD,
    }

def to_base64(img: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def save_upload(file_bytes: bytes, original_name: str) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_path = UPLOAD_DIR / f"{timestamp}_{original_name.replace(' ', '_')}"
    with open(save_path, "wb") as f:
        f.write(file_bytes)
    return save_path

def save_result_image(img: np.ndarray, prefix: str, original_name: str) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    unique_name = f"{timestamp}_{prefix}_{Path(original_name).stem}.png"
    save_path = UPLOAD_DIR / unique_name
    Image.fromarray(img).save(save_path)
    return save_path

# ---------------- ROUTES ----------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "Doctor")
        session["user"] = username
        return redirect(url_for("index"))
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

@app.route("/")
def index():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("index.html", username=session["user"])

@app.route("/predict", methods=["POST"])
def predict_route():
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        file_bytes = file.read()

        # Load Image
        img_rgb = load_image(file_bytes, file.filename)

        # --- SMART VALIDATION GUARD ---
        is_valid, error_msg = validate_image(img_rgb, file.filename)
        if not is_valid:
            return jsonify({"error": error_msg}), 400
        # ------------------------------

        # Save and Predict
        saved_path = save_upload(file_bytes, file.filename)
        result = predict(img_rgb)

        original_b64 = to_base64(img_rgb)
        overlay = result.pop("gradcam_overlay")
        gradcam_b64 = to_base64(overlay)

        save_result_image(img_rgb, "original", file.filename)
        save_result_image(overlay, "gradcam", file.filename)

        result["original_image"] = f"data:image/png;base64,{original_b64}"
        result["gradcam_image"] = f"data:image/png;base64,{gradcam_b64}"
        result["filename"] = file.filename
        return jsonify(result)

    except Exception as e:
        print(f"ERROR: {e}")
        return jsonify({"error": "Failed to process image. The file may be corrupted."}), 500

@app.route("/health")
def health():
    return {"status": "ok", "device": str(DEVICE), "threshold": OPTIMAL_THRESHOLD}

if __name__ == "__main__":
    print("=" * 60)
    print("Pneumonia Detection Web UI (Smart Validation)")
    print("=" * 60)
    load_model()
    app.run(host="0.0.0.0", port=5000, debug=False)