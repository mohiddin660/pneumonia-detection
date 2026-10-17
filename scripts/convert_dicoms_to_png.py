"""
Convert all RSNA DICOM files to PNG for faster training.

DICOM loading is 10x slower than PNG — converting once
speeds up every training epoch dramatically.

Run:  python -m scripts.convert_dicoms_to_png
"""
from pathlib import Path
import time

import cv2
import numpy as np
import pydicom
from tqdm import tqdm

SRC = Path("data/raw/stage_2_train_images")
DST = Path("data/processed/dicom_as_png")
DST.mkdir(parents=True, exist_ok=True)


def dicom_to_uint8(path: Path) -> np.ndarray:
    """Load a DICOM and return a normalized uint8 array."""
    ds = pydicom.dcmread(str(path))
    pixels = ds.pixel_array.astype(np.float32)
    pixels = pixels - pixels.min()
    if pixels.max() > 0:
        pixels = pixels / pixels.max() * 255.0
    return pixels.astype(np.uint8)


def main():
    dcm_files = sorted(SRC.glob("*.dcm"))
    total = len(dcm_files)

    if total == 0:
        print(f"ERROR: No .dcm files found in {SRC}")
        return

    print("=" * 60)
    print(f"Converting {total} DICOMs to PNG")
    print(f"Source: {SRC}")
    print(f"Dest:   {DST}")
    print("=" * 60)

    converted = 0
    skipped = 0
    failed = 0

    t0 = time.time()

    for dcm in tqdm(dcm_files, desc="Converting"):
        png_path = DST / f"{dcm.stem}.png"

        if png_path.exists():
            skipped += 1
            continue

        try:
            img = dicom_to_uint8(dcm)
            cv2.imwrite(str(png_path), img)
            converted += 1
        except Exception as e:
            print(f"\nFailed {dcm.name}: {e}")
            failed += 1

    elapsed = time.time() - t0

    print("\n" + "=" * 60)
    print(f"[OK] Converted: {converted}")
    print(f"     Skipped:   {skipped} (already existed)")
    print(f"     Failed:    {failed}")
    print(f"     Time:      {elapsed:.1f} seconds")
    print(f"     Location:  {DST}")
    print("=" * 60)


if __name__ == "__main__":
    main()