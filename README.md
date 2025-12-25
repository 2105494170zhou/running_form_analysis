# Project Overview

A video-based running form analysis system that:
1) extracts pose keypoints from a **side-view running video** (MediaPipe),
2) computes **kinematic running metrics**,
3) runs **ML classifiers** to flag common form issues,
4) returns a JSON report **plus an overlay video** with pose landmarks.
5) Public website powered by **Framer** frontend and **Hugging Face** backend program: [runningform.framer.website](https://runningform.framer.website)


---

## What it does

### Metrics (computed from pose)
The API returns a `_metrics` object with:
- `cadence_spm` (steps/min)
- `stride_length_mean_m` (meters)
- `vertical_oscillation_cm` (cm)
- `forward_lean_mean_deg` (deg)
- `avg_shin_angle_IC_deg` (deg, at initial contact)
- `avg_knee_flex_IC_deg` (deg, at initial contact)
- `peak_knee_flex_deg` (deg)
- `avg_foot_dorsiflex_wrt_horiz_IC_deg` (deg, at initial contact)

### ML form issue labels (0/1)
The API predicts 4 technique issues:
- `overstriding`
- `hard_landing`
- `forward_lean`
- `little_propulsion`

### Model Training
The machines learning models were trained on 100+ videos from the runner on the Northfield Mount Hermon Cross-Country team to best represent the teenager running form. Currently the size of the dataset is still expanding.
