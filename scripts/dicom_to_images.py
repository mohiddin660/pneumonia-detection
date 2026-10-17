"""
Convert DICOM X-rays to viewable PNG images.

Run:  python -m scripts.dicom_to_images

Creates:
    reports/dicom_previews/sample_grid.png    (4 normal + 4 pneumonia)
    reports/dicom_previews/single_XXXX.png    (one detailed example)
    data/processed/images/*.png               (500 exported PNGs)
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---------------- CONFIG ----------------
DICOM_DIR = Path("data/raw/stage_2_train_images")
LABELS_CSV = Path("data/raw/stage_2_train_labels.csv")
OUT_PREVIEW = Path("reports/dicom_previews")
OUT_IMAGES = Path("data/processed/images")

OUT_PREVIEW.mkdir(parents=True, exist_ok=True)
OUT_IMAGES.mkdir(parents=True, exist_ok=True)


def dicom_to_array(dicom_path: str) -> np.ndarray:
    """Load a DICOM file and return a normalized uint8 image."""
    ds = pydicom.dcmread(dicom_path)
    pixels = ds.pixel_array.astype(np.float32)

    # Normalize to 0-255
    pixels = pixels - pixels.min()
    if pixels.max() > 0:
        pixels = pixels / pixels.max() * 255.0

    return pixels.astype(np.uint8)


def draw_bboxes(img: np.ndarray, boxes: pd.DataFrame) -> np.ndarray:
    """Draw red bounding boxes on a copy of the image."""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for _, row in boxes.iterrows():
        if pd.notna(row["x"]) and row["width"] > 0 and row["height"] > 0:
            x, y = int(row["x"]), int(row["y"])
            w, h = int(row["width"]), int(row["height"])
            cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (255, 0, 0), 4)
    return img_rgb


def create_preview_grid(csv_path: Path, num_normal: int = 4, num_pneumonia: int = 4):
    """Create a grid: 4 normal + 4 pneumonia images with labels."""
    labels = pd.read_csv(csv_path)
    labels["Target"] = labels["Target"].astype(int)

    normals = labels[labels["Target"] == 0]["patientId"].unique()[:num_normal]
    pneumonias = labels[labels["Target"] == 1]["patientId"].unique()[:num_pneumonia]

    fig, axes = plt.subplots(2, 4, figsize=(16, 9))

    # Row 1: NORMAL
    for i, pid in enumerate(normals):
        dcm_path = DICOM_DIR / f"{pid}.dcm"
        if not dcm_path.exists():
            axes[0, i].set_visible(False)
            continue
        img = dicom_to_array(str(dcm_path))
        axes[0, i].imshow(img, cmap="gray")
        axes[0, i].set_title(f"NORMAL\n{pid[:8]}...", fontsize=11, color="green", fontweight="bold")
        axes[0, i].axis("off")

    # Row 2: PNEUMONIA (with bounding boxes)
    for i, pid in enumerate(pneumonias):
        dcm_path = DICOM_DIR / f"{pid}.dcm"
        if not dcm_path.exists():
            axes[1, i].set_visible(False)
            continue
        img = dicom_to_array(str(dcm_path))
        boxes = labels[labels["patientId"] == pid]
        img_with_boxes = draw_bboxes(img, boxes)
        axes[1, i].imshow(img_with_boxes)
        axes[1, i].set_title(f"PNEUMONIA\n{pid[:8]}...", fontsize=11, color="red", fontweight="bold")
        axes[1, i].axis("off")

    plt.suptitle("Sample X-rays from RSNA Dataset  (Red boxes = annotated pneumonia regions)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    out = OUT_PREVIEW / "sample_grid.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved grid: {out}")


def export_all_images(csv_path: Path, max_images: int = 500):
    """Export first N DICOMs to PNG for inspection."""
    labels = pd.read_csv(csv_path)
    unique_pids = labels["patientId"].unique()[:max_images]

    count = 0
    for pid in tqdm(unique_pids, desc="Exporting images"):
        dcm_path = DICOM_DIR / f"{pid}.dcm"
        if not dcm_path.exists():
            continue
        try:
            img = dicom_to_array(str(dcm_path))
            out_path = OUT_IMAGES / f"{pid}.png"
            cv2.imwrite(str(out_path), img)
            count += 1
        except Exception as e:
            print(f"Skip {pid}: {e}")

    print(f"[OK] Exported {count} images to {OUT_IMAGES}")


def show_single_image(pid: str):
    """Show one DICOM with metadata + bounding boxes."""
    dcm_path = DICOM_DIR / f"{pid}.dcm"
    ds = pydicom.dcmread(str(dcm_path))
    img = dicom_to_array(str(dcm_path))

    # Load boxes
    labels = pd.read_csv(LABELS_CSV)
    boxes = labels[labels["patientId"] == pid]

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Left: plain
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Raw X-ray", fontsize=12)
    axes[0].axis("off")

    # Right: with boxes
    if boxes["width"].notna().any():
        img_with_boxes = draw_bboxes(img, boxes)
        axes[1].imshow(img_with_boxes)
        axes[1].set_title(f"With bounding boxes ({len(boxes)} region(s))", fontsize=12)
    else:
        axes[1].imshow(img, cmap="gray")
        axes[1].set_title("No bounding boxes (NORMAL)", fontsize=12)
    axes[1].axis("off")

    plt.suptitle(
        f"Patient: {pid}\n"
        f"Image size: {img.shape}  |  "
        f"Age: {getattr(ds, 'PatientAge', 'N/A')}  |  "
        f"Sex: {getattr(ds, 'PatientSex', 'N/A')}  |  "
        f"View: {getattr(ds, 'ViewPosition', 'N/A')}",
        fontsize=11,
    )
    plt.tight_layout()

    out = OUT_PREVIEW / f"single_{pid[:8]}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {out}")


def main():
    print("=" * 60)
    print("DICOM to Image Converter")
    print("=" * 60)

    # Step 1
    print("\n[1/3] Creating sample grid (4 normal + 4 pneumonia)...")
    create_preview_grid(LABELS_CSV)

    # Step 2
    print("\n[2/3] Exporting 500 sample images as PNG...")
    export_all_images(LABELS_CSV, max_images=500)

    # Step 3
    print("\n[3/3] Showing one detailed example...")
    labels = pd.read_csv(LABELS_CSV)
    sample_pid = labels[labels["Target"] == 1]["patientId"].iloc[0]
    show_single_image(sample_pid)

    print("\n" + "=" * 60)
    print("[DONE]")
    print(f"Preview images: {OUT_PREVIEW}")
    print(f"Exported PNGs:  {OUT_IMAGES}")
    print("=" * 60)


if __name__ == "__main__":
    main()