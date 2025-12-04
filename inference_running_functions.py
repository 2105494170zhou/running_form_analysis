#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Inference script for running-form classification using 4 separate ML models
(one per label).

The models can be of different types (e.g. RandomForest, XGBoost, etc.) as long
as they implement a scikit-learn style API with .predict_proba().

Usage:
    python inference_running.py --video "D:\\path\\to\\video.mp4"

Requirements:
    pip install mediapipe opencv-python pandas numpy scikit-learn xgboost joblib
"""

import argparse
from pathlib import Path
from typing import Dict, Tuple, List, Any

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb  # needed so XGB models can be unpickled, even if not used directly


# ===============================
# MediaPipe pose extraction
# ===============================

mp_pose = mp.solutions.pose


def extract_pose_from_video(video_path: Path) -> pd.DataFrame:
    """
    Run MediaPipe Pose on a video and return a long-format DataFrame:
        columns: frame_index, landmark_id, x_norm, y_norm, z_norm, visibility

    Each row corresponds to ONE landmark in ONE frame.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    rows = []
    frame_idx = 0

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(image_rgb)

            if result.pose_landmarks:
                # We have a pose detection for this frame
                for lm_id, lm in enumerate(result.pose_landmarks.landmark):
                    rows.append({
                        "frame_index": frame_idx,
                        "landmark_id": lm_id,
                        "x_norm": lm.x,
                        "y_norm": lm.y,
                        "z_norm": lm.z,
                        "visibility": lm.visibility,
                    })
            else:
                # No detection in this frame: fill with NaN so we know it's missing
                for lm_id in range(33):  # MediaPipe Pose has 33 landmarks
                    rows.append({
                        "frame_index": frame_idx,
                        "landmark_id": lm_id,
                        "x_norm": np.nan,
                        "y_norm": np.nan,
                        "z_norm": np.nan,
                        "visibility": np.nan,
                    })

            frame_idx += 1

    cap.release()
    df = pd.DataFrame(rows)
    return df


# ===============================
# Preprocessing: pivot + cleaning + frame mask
# ===============================

def df_to_sequence_and_mask(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a MediaPipe pose DataFrame into:
        seq:        [T, F] float32 (no NaNs, NaNs replaced by 0)
        frame_mask: [T]   float32, 1 = frame has some real data, 0 = no data

    Steps:
      - Check required columns.
      - Pivot (frame_index, landmark_id) -> frames x (coords x landmarks).
      - Replace NaNs with 0.0 but keep where frames were all-NaN via frame_mask.
    """
    if "frame_index" not in df.columns or "landmark_id" not in df.columns:
        raise ValueError("DataFrame must contain 'frame_index' and 'landmark_id' columns.")

    coord_cols = [c for c in ["x_norm", "y_norm", "z_norm", "visibility"] if c in df.columns]
    if not coord_cols:
        raise ValueError("No coordinate columns found in DataFrame.")

    # Sort and pivot to ensure consistent ordering: frames x features
    df_sorted = df.sort_values(["frame_index", "landmark_id"])
    pivot = df_sorted.pivot_table(index="frame_index", columns="landmark_id", values=coord_cols)

    # Flatten MultiIndex columns to 'x_norm_lm0', 'y_norm_lm0', etc.
    pivot.columns = [f"{coord}_lm{lm}" for coord, lm in pivot.columns]

    seq = pivot.to_numpy(dtype="float32")  # [T, F], may contain NaNs

    # frame_mask: 1 if this frame has at least one non-NaN value, else 0
    frame_has_data = ~np.isnan(seq).all(axis=1)   # [T] bool
    frame_mask = frame_has_data.astype("float32")

    # Replace NaNs with 0 so no NaNs go into the model
    seq = np.where(np.isnan(seq), 0.0, seq).astype("float32")

    return seq, frame_mask


# ===============================
# Load 4 separate models (any type)
# ===============================

def load_models_per_label(model_dir: Path,
                          label_names: List[str]) -> Dict[str, Any]:
    """
    Load one ML model per label.

    The models can be different types (RandomForest, XGBoost, etc.) as long as
    they support .predict_proba() with the scikit-learn API.

    Assumes filenames like:
        rf_overstriding.pkl
        rf_hard_landing.pkl
        rf_forward_lean.pkl
        xgb_little_propulsion.pkl
    """
    models: Dict[str, Any] = {}

    # Adjust names here if your filenames differ.
    # Keys in this dict must match LABEL_NAMES below.
    model_paths = {
        "overstriding":      model_dir / "xgb_overstriding.pkl",
        "hard_landing":      model_dir / "rf_hard_landing.pkl",
        "forward_lean":      model_dir / "rf_forward_lean.pkl",
        "little_propulsion": model_dir / "xgb_little_propulsion.pkl",
    }

    for label in label_names:
        path = model_paths[label]
        if not path.exists():
            raise FileNotFoundError(f"Model file for label '{label}' not found: {path}")
        print(f"Loading model for '{label}' from: {path}")
        models[label] = joblib.load(path)

    return models


# ===============================
# Inference on a single video
# ===============================

def predict_video(
    video_path: Path,
    models_per_label: Dict[str, Any],
    label_names: List[str],
    feature_dim: int,
) -> Tuple[Dict[str, float], Dict[str, int]]:
    """
    Full pipeline for a single video:
      - MediaPipe pose -> DataFrame
      - DataFrame -> (seq, frame_mask)
      - seq/mask -> masked-mean feature vector [1, F]
      - Run each label's model separately -> probability + binary prediction

    Returns:
      probs: {label_name -> probability of class 1}
      preds: {label_name -> 0 or 1}
    """
    print(f"\nProcessing video: {video_path}")
    df_pose = extract_pose_from_video(video_path)
    num_frames = df_pose["frame_index"].nunique()
    print(f"Extracted pose for {num_frames} frames.")

    seq, frame_mask = df_to_sequence_and_mask(df_pose)
    T, F = seq.shape
    print(f"Sequence shape: [T={T}, F={F}]")

    if F != feature_dim:
        raise ValueError(f"Feature dim mismatch: got {F}, expected {feature_dim}.")

    # ----- Build masked-mean feature vector (same as training X_features) -----
    mask = frame_mask.astype("float32")[:, np.newaxis]  # [T, 1]

    # Multiply each frame by its mask (0 or 1) and sum over time
    X_sum = (seq * mask).sum(axis=0)  # [F]
    valid_len = mask.sum()
    if valid_len <= 0:
        # Fallback: simple mean over time if mask somehow all zeros
        feats = seq.mean(axis=0)
    else:
        feats = X_sum / valid_len  # [F]

    # Model expects shape [N, F] even for a single example
    X_input = feats.reshape(1, -1)  # [1, F]

    probs: Dict[str, float] = {}
    preds: Dict[str, int] = {}

    for name in label_names:
        model = models_per_label[name]

        # We assume a scikit-learn style API: predict_proba returns [N, n_classes]
        proba = model.predict_proba(X_input)[0]  # first (and only) sample

        if proba.shape[0] == 2:
            # Standard binary case: [P(class 0), P(class 1)]
            p1 = float(proba[1])   # probability of class 1
        else:
            # Degenerate / unusual case:
            # If there's only one class, scikit-like models often return [1.0].
            # Here we conservatively treat this as prob=0 for class 1.
            p1 = 0.0

        probs[name] = p1
        preds[name] = int(p1 >= 0.5)

    return probs, preds


# ===============================
# High-level helper for web API
# ===============================

# These are the label names used inside your code and mapping above
LABEL_NAMES = ["overstriding", "hard_landing", "forward_lean", "little_propulsion"]

# 33 MediaPipe landmarks * 4 values (x, y, z, visibility) = 132 features
FEATURE_DIM = 132

# Where the .pkl model files are located.
# In your case, the 4 models are in the SAME folder as this script,
# so we point to this file's directory.
MODEL_DIR = Path(__file__).resolve().parent

# Load the models ONCE when this module is imported.
# This way, Flask doesn't have to reload all models on every request.
MODELS_PER_LABEL = load_models_per_label(MODEL_DIR, LABEL_NAMES)


def run_inference_on_video(video_path: Path):
    """
    High-level helper used by the web API or CLI.

    Input:
        video_path: Path to a video file on disk.

    Behavior:
        - Runs the full pipeline:
            MediaPipe pose extraction
            -> DataFrame -> sequence/mask
            -> masked mean pooling -> feature vector [1, FEATURE_DIM]
            -> runs each of the 4 models

    Returns:
        probs: {label_name: probability of issue (class 1)}
        preds: {label_name: 0 or 1}
    """
    probs, preds = predict_video(
        video_path=video_path,
        models_per_label=MODELS_PER_LABEL,
        label_names=LABEL_NAMES,
        feature_dim=FEATURE_DIM,
    )
    return probs, preds
