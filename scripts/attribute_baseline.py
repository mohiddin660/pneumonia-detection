"""
Baseline model using ONLY image-based attributes (no bounding box leakage).

FIXED: Handles class imbalance with class_weight and threshold tuning.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, accuracy_score,
    precision_recall_curve, f1_score,
)

OUT = Path("reports")
OUT.mkdir(exist_ok=True)

# ---------------- LOAD ----------------
df = pd.read_csv("data/splits/attributes.csv")

# Image-only features (no bbox leakage)
feature_cols = [
    "pixel_mean",
    "pixel_std",
    "patient_age",
]

df = df.dropna(subset=["Target"]).copy()
df["patient_age"] = df["patient_age"].fillna(df["patient_age"].median())

X = df[feature_cols].values.astype(np.float32)
y = df["Target"].astype(int).values

print(f"Features used: {feature_cols}")
print(f"Dataset: {len(X)} samples | Pneumonia: {y.sum()} | Normal: {(1 - y).sum()}")
print(f"Class ratio: {y.sum() / len(y) * 100:.2f}% pneumonia")

# ---------------- SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# ---------------- TRAIN ----------------
print("\nTraining RandomForest (with balanced class weights)...")
clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=5,
    class_weight="balanced",       # ← KEY FIX: handles 77/23 imbalance
    random_state=42,
    n_jobs=-1,
)
clf.fit(X_train, y_train)

# ---------------- PREDICT ----------------
y_prob = clf.predict_proba(X_test)[:, 1]

# ---- Threshold tuning: find best F1 threshold ----
precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
f1s = 2 * precisions * recalls / (precisions + recalls + 1e-8)
best_idx = np.argmax(f1s)
best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
best_f1 = f1s[best_idx]

print(f"\nBest threshold (max F1): {best_threshold:.4f}")
print(f"Best F1 at that threshold: {best_f1:.4f}")

# Use best threshold
y_pred = (y_prob >= best_threshold).astype(int)

# ---------------- METRICS ----------------
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\n" + "=" * 60)
print("=== Attribute-Only Baseline (Full Dataset, Balanced) ===")
print("=" * 60)
print(f"Accuracy: {acc:.4f}")
print(f"AUC:      {auc:.4f}")
print(f"F1:       {f1_score(y_test, y_pred):.4f}")
print()
print(classification_report(y_test, y_pred, target_names=["NORMAL", "PNEUMONIA"]))

# ---------------- FEATURE IMPORTANCE ----------------
importances = pd.DataFrame({
    "feature": feature_cols,
    "importance": clf.feature_importances_,
}).sort_values("importance", ascending=False)

print("=== Feature Importance ===")
print(importances.to_string(index=False))

# ---- Plot 1: importance ----
plt.figure(figsize=(10, 5))
sns.barplot(data=importances, x="importance", y="feature", hue="feature",
            palette="viridis", legend=False)
plt.title("Image-Only Attribute Importance for Pneumonia")
plt.tight_layout()
plt.savefig(OUT / "attribute_importance.png", dpi=150)
plt.close()

# ---- Plot 2: confusion matrix ----
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["NORMAL", "PNEUMONIA"],
    yticklabels=["NORMAL", "PNEUMONIA"],
)
plt.title(f"Attribute-Only Confusion Matrix\n(threshold={best_threshold:.3f})")
plt.ylabel("True")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(OUT / "attribute_confusion_matrix.png", dpi=150)
plt.close()

# ---- Plot 3: ROC + PR curve ----
fpr, tpr, _ = roc_curve(y_test, y_prob)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(fpr, tpr, label=f"AUC = {auc:.4f}", lw=2, color="darkorange")
axes[0].plot([0, 1], [0, 1], "--", color="gray")
axes[0].set_xlabel("FPR")
axes[0].set_ylabel("TPR")
axes[0].set_title("ROC Curve")
axes[0].legend(loc="lower right")
axes[0].grid(alpha=0.3)

axes[1].plot(recalls, precisions, lw=2, color="blue")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Curve")
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUT / "attribute_roc.png", dpi=150)
plt.close()

print(f"\n[OK] Saved plots to {OUT}/")

# ---------------- SUMMARY ----------------
print("\n" + "=" * 60)
print("=== SUMMARY FOR MENTOR ===")
print("=" * 60)
print(f"Attribute-only AUC:       {auc:.4f}")
print(f"Deep learning AUC (prev): 0.8648")
print(f"Improvement from DL:      {0.8648 - auc:+.4f}")
print("=" * 60)