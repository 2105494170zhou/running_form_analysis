
import json
import tempfile
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from huggingface_hub import hf_hub_download
from mediapipe.framework.formats import landmark_pb2

from running_metrics_extraction_4 import process_single_csv



# Constants


LABEL_NAMES = ["overstriding", "hard_landing", "forward_lean", "little_propulsion"]


FEATURE_DIM = 132

MODEL_FILENAMES = {
    "overstriding": "xgb_overstriding.pkl",
    "hard_landing": "rf_hard_landing.pkl",
    "forward_lean": "rf_forward_lean.pkl",
    "little_propulsion": "xgb_little_propulsion.pkl",
}

_MODELS_CACHE = {}
_LOADED_REPO_ID = None

# MediaPipe setup


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


def extract_pose_from_video(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    rows = []
    frame_idx = 0

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as pose:

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)

            if result.pose_landmarks:
                for lm_id, lm in enumerate(result.pose_landmarks.landmark):
                    rows.append(
                        {
                            "frame_index": frame_idx,
                            "landmark_id": lm_id,
                            "x_norm": lm.x,
                            "y_norm": lm.y,
                            "z_norm": lm.z,
                            "visibility": lm.visibility,
                            "x_px": lm.x * w,
                            "y_px": lm.y * h,
                        }
                    )
            else:
                for lm_id in range(33):
                    rows.append(
                        {
                            "frame_index": frame_idx,
                            "landmark_id": lm_id,
                            "x_norm": np.nan,
                            "y_norm": np.nan,
                            "z_norm": np.nan,
                            "visibility": np.nan,
                            "x_px": np.nan,
                            "y_px": np.nan,
                        }
                    )

            frame_idx += 1

    cap.release()
    return pd.DataFrame(rows)


def generate_overlay_video_from_df(video_path, output_path, df_pose, scale=0.7):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    out_w = max(2, int(orig_w * scale))
    out_h = max(2, int(orig_h * scale))
    out_w -= out_w % 2
    out_h -= out_h % 2

    print(f"[OVERLAY] {orig_w}x{orig_h} -> {out_w}x{out_h}", flush=True)

    fourcc = cv2.VideoWriter_fourcc(*"VP80")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, out_h))
    if not out.isOpened():
        raise RuntimeError(f"Could not open VP8 VideoWriter for {output_path}")

    frame_idx = 0
    total = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_small = cv2.resize(frame, (out_w, out_h))

        frame_df = df_pose[df_pose["frame_index"] == frame_idx]
        if not frame_df.empty and not frame_df[["x_norm", "y_norm"]].isna().all().all():
            lms = []
            for _, row in frame_df.sort_values("landmark_id").iterrows():
                if np.isnan(row["x_norm"]) or np.isnan(row["y_norm"]):
                    continue
                lm = landmark_pb2.NormalizedLandmark(
                    x=float(row["x_norm"]),
                    y=float(row["y_norm"]),
                    z=float(row["z_norm"]) if not np.isnan(row["z_norm"]) else 0.0,
                    visibility=float(row["visibility"]) if not np.isnan(row["visibility"]) else 0.0,
                )
                lms.append(lm)

            if lms:
                landmark_list = landmark_pb2.NormalizedLandmarkList(landmark=lms)
                mp_drawing.draw_landmarks(
                    frame_small,
                    landmark_list,
                    mp_pose.POSE_CONNECTIONS,
                    mp_styles.get_default_pose_landmarks_style(),
                )

        out.write(frame_small)
        frame_idx += 1
        total += 1

    cap.release()
    out.release()

    print(f"[OVERLAY] Wrote {total} frames at {fps} fps -> {output_path}", flush=True)
    return total, fps


def df_to_sequence_and_mask(df):
    if "frame_index" not in df.columns or "landmark_id" not in df.columns:
        raise ValueError("DataFrame must contain 'frame_index' and 'landmark_id'.")

    coord_cols = [c for c in ["x_norm", "y_norm", "z_norm", "visibility"] if c in df.columns]
    if not coord_cols:
        raise ValueError("No coordinate columns found in DataFrame.")

    df_sorted = df.sort_values(["frame_index", "landmark_id"])
    pivot = df_sorted.pivot_table(index="frame_index", columns="landmark_id", values=coord_cols)

    pivot.columns = [f"{coord}_lm{lm}" for coord, lm in pivot.columns]

    seq = pivot.to_numpy(dtype="float32")

    mask = (~np.isnan(seq).all(axis=1)).astype("float32")
    seq = np.where(np.isnan(seq), 0.0, seq).astype("float32")

    return seq, mask


def load_models_from_hub(model_repo):

    global _MODELS_CACHE, _LOADED_REPO_ID

    if _MODELS_CACHE and _LOADED_REPO_ID == model_repo:
        return _MODELS_CACHE

    models = {}
    for label in LABEL_NAMES:
        filename = MODEL_FILENAMES[label]
        print(f"[MODEL] Downloading {model_repo}/{filename}", flush=True)
        local_path = hf_hub_download(repo_id=model_repo, filename=filename)
        models[label] = joblib.load(local_path)

    _MODELS_CACHE = models
    _LOADED_REPO_ID = model_repo
    return models


def predict_video(video_path, models_per_label, df_pose=None):

    print(f"\n[INFER] Processing video: {video_path}", flush=True)

    if df_pose is None:
        df_pose = extract_pose_from_video(video_path)

    num_frames = df_pose["frame_index"].nunique()
    print(f"[INFER] Pose extracted for {num_frames} frames", flush=True)

    seq, frame_mask = df_to_sequence_and_mask(df_pose)
    T, F = seq.shape
    print(f"[INFER] Sequence shape: T={T}, F={F}", flush=True)

    if F != FEATURE_DIM:
        raise ValueError(f"Feature dim mismatch: got {F}, expected {FEATURE_DIM}")

    # masked mean over time
    mask = frame_mask[:, None]
    valid = mask.sum()
    if valid <= 0:
        feats = seq.mean(axis=0)
    else:
        feats = (seq * mask).sum(axis=0) / valid

    X = feats.reshape(1, -1)

    probs = {}
    preds = {}

    for label in LABEL_NAMES:
        model = models_per_label[label]
        proba = model.predict_proba(X)[0]

        p1 = float(proba[1]) if len(proba) == 2 else 0.0
        probs[label] = p1
        preds[label] = int(p1 >= 0.5)

    return probs, preds


def compute_metrics_for_df(df_pose, fps, height_m=1.70):

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        csv_path = tmpdir / "pose_points.csv"

        df_pose.to_csv(csv_path, index=False)

        json_path = process_single_csv(
            csv_path=csv_path,
            fps=fps,
            height_m=height_m,
            coords="norm",
            vel_thresh=0.0175,
            vel_thresh_pct=0.0,
            persist_frames=8,
            min_step_gap=19,
            smooth_alpha=0.25,
            max_interp_gap=3,
            view="side",
            out_dir=tmpdir,
        )

        if json_path is None:
            return {}

        with open(json_path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        return obj.get("metrics", {})


def run_inference_and_overlay(video_path, model_repo, overlay_path, scale=0.7, height_m=1.70):

    models = load_models_from_hub(model_repo)

    df_pose = extract_pose_from_video(video_path)
    frame_count = df_pose["frame_index"].nunique()
    print(f"[PIPELINE] Extracted pose for {frame_count} frames", flush=True)

    probs, preds = predict_video(
        video_path=video_path,
        models_per_label=models,
        df_pose=df_pose,
    )

    _, fps = generate_overlay_video_from_df(
        video_path=video_path,
        output_path=overlay_path,
        df_pose=df_pose,
        scale=scale,
    )

    metrics = compute_metrics_for_df(df_pose=df_pose, fps=fps, height_m=height_m)

    return probs, preds, frame_count, fps, metrics


def run_inference_on_video(video_path, model_repo):
    models = load_models_from_hub(model_repo)
    return predict_video(video_path=video_path, models_per_label=models)
