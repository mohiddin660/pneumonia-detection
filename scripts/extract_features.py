"""
Explicit feature extraction using the trained EfficientNet backbone.

Saves 1280-dim deep features + attributes to CSV.

Run:  python -m scripts.extract_features
Output: data/splits/features.csv
"""
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import RSNAPneumoniaDataset
from src.data.transforms import get_eval_transforms
from src.models.classifier import PneumoniaClassifier


def extract_deep(model, loader, device):
    """Extract 1280-dim features from backbone for all batches."""
    feats = []
    model.eval()
    with torch.no_grad():
        for imgs, _, _ in tqdm(loader, desc="Extracting deep features"):
            imgs = imgs.to(device)
            f = model.backbone(imgs).cpu().numpy()
            feats.append(f)
    return np.concatenate(feats, axis=0)


def main():
    OUT = Path("data/splits/features.csv")
    OUT.parent.mkdir(exist_ok=True, parents=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print("Loading trained model...")
    model = PneumoniaClassifier(multi_task=True).to(device)
    ckpt = torch.load("models/best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Load data
    print("Loading dataset...")
    ds = RSNAPneumoniaDataset(
        "data/splits/train.csv",
        get_eval_transforms(224),
    )
    # num_workers=0 for Windows compatibility in this script
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    # Extract
    print(f"Extracting features from {len(ds)} images...")
    deep_feats = extract_deep(model, loader, device)
    print(f"Deep feature matrix shape: {deep_feats.shape}")

    # Labels
    labels = ds.df[["patientId", "Target", "stage"]].reset_index(drop=True)

    # Build DataFrame
    feat_cols = [f"f{i}" for i in range(deep_feats.shape[1])]
    deep_df = pd.DataFrame(deep_feats, columns=feat_cols)
    merged = pd.concat([labels, deep_df], axis=1)

    # Save
    merged.to_csv(OUT, index=False)
    print(f"[OK] Saved {merged.shape[0]} samples x {merged.shape[1]} columns to {OUT}")


if __name__ == "__main__":
    main()