"""
Prepare train/val/test CSVs for RSNA pneumonia detection.

Run:  python -m scripts.prepare_data

Output:
    data/splits/train.csv
    data/splits/val.csv
    data/splits/test.csv

Each CSV has columns:
    patientId, image_path, Target, stage
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.stage_labeler import compute_stages_from_labels
from src.utils.seed import set_seed
from src.utils.logger import get_logger

log = get_logger("prepare_data", "logs/prepare_data.log")

# ---------------- CONFIG ----------------
LABELS_CSV = "data/raw/stage_2_train_labels.csv"
IMAGES_DIR = Path("data/raw/stage_2_train_images")
OUT_DIR = Path("data/splits")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42


# ---------------- MAIN ----------------
def main():
    set_seed(SEED)

    log.info("=" * 60)
    log.info("Preparing RSNA data splits")
    log.info("=" * 60)

    # 1. Compute stage labels from bounding boxes
    log.info("Computing stage labels from bounding boxes...")
    stage_df = compute_stages_from_labels(LABELS_CSV)
    log.info(f"  → {len(stage_df)} unique patients")

    # 2. Build image_path
    log.info("Building image paths...")
    stage_df["image_path"] = stage_df["patientId"].apply(
        lambda pid: str(IMAGES_DIR / f"{pid}.dcm")
    )

    # 3. Verify all images exist
    log.info("Verifying DICOM files exist...")
    missing = 0
    for p in stage_df["image_path"]:
        if not Path(p).exists():
            missing += 1
    if missing > 0:
        log.warning(f"  ⚠️ {missing} images missing on disk!")
    else:
        log.info(f"  ✅ All {len(stage_df)} DICOMs found on disk")

    # 4. Keep only needed columns
    df = stage_df[["patientId", "image_path", "Target", "stage"]].copy()

    # 5. Summary before splitting
    log.info("\nDataset summary:")
    log.info(f"  Total: {len(df)}")
    log.info(f"  Normal (Target=0): {(df['Target'] == 0).sum()}")
    log.info(f"  Pneumonia (Target=1): {(df['Target'] == 1).sum()}")
    pneumonia = df[df["Target"] == 1]
    for stage_id, name in [(0, "MILD"), (1, "MODERATE"), (2, "SEVERE")]:
        log.info(f"    {name}: {(pneumonia['stage'] == stage_id).sum()}")

    # 6. Split: stratify by Target so class balance is preserved
    log.info("\nSplitting data...")
    train_df, temp_df = train_test_split(
        df,
        test_size=(VAL_RATIO + TEST_RATIO),
        stratify=df["Target"],
        random_state=SEED,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
        stratify=temp_df["Target"],
        random_state=SEED,
    )

    # 7. Save
    train_df.to_csv(OUT_DIR / "train.csv", index=False)
    val_df.to_csv(OUT_DIR / "val.csv", index=False)
    test_df.to_csv(OUT_DIR / "test.csv", index=False)

    log.info("\n✅ Splits saved:")
    log.info(f"  Train: {len(train_df)}")
    log.info(f"  Val:   {len(val_df)}")
    log.info(f"  Test:  {len(test_df)}")
    log.info(f"  → {OUT_DIR}/")


if __name__ == "__main__":
    main()