
import os
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename

from inference_running_functions import LABEL_NAMES, run_inference_and_overlay

# Config
MODEL_REPO = os.getenv("HF_MODEL_REPO", "adkjfbskd/running_form_model")

UPLOAD_DIR = Path("/tmp/videos")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

OVERLAY_DIR = Path("/tmp/overlays")
OVERLAY_DIR.mkdir(parents=True, exist_ok=True)


# App setup

app = Flask(__name__)

CORS(
    app,
    resources={r"/*": {"origins": "*"}},
    supports_credentials=False,
)


# Routes

@app.route("/health", methods=["GET"])
@cross_origin()
def health():
    return jsonify({"status": "ok"})


@app.route("/", methods=["GET"])
@cross_origin()
def index():
    return jsonify({"message": "Running Form API is alive. Use POST /analyze with a 'video' file."})


@app.route("/analyze", methods=["POST"])
@cross_origin()
def analyze():
    if "video" not in request.files:
        return jsonify({"error": "No file field named 'video' in the request."}), 400

    file = request.files["video"]
    if not file.filename:
        return jsonify({"error": "Uploaded file has an empty filename."}), 400

    filename = secure_filename(file.filename)
    save_path = UPLOAD_DIR / filename
    file.save(save_path)

    height_m = 1.70
    height_str = request.form.get("height_m")
    if height_str is not None:
        try:
            height_m = float(height_str)
        except ValueError:
            pass 

    # Output overlay (WebM)
    overlay_name = f"{save_path.stem}_overlay.webm"
    overlay_path = OVERLAY_DIR / overlay_name

    try:
        probs, preds, frame_count, fps, metrics = run_inference_and_overlay(
            video_path=save_path,
            model_repo=MODEL_REPO,
            overlay_path=overlay_path,
            scale=0.7,
            height_m=height_m,
        )
    except Exception as e:
        print(f"[ERROR] Inference/overlay failed for {save_path}: {e}", flush=True)
        return jsonify({"error": f"Inference failed: {str(e)}"}), 500
    finally:
        try:
            os.remove(save_path)
        except OSError:
            pass

    result = {}
    for name in LABEL_NAMES:
        result[name] = {"prob": float(probs[name]), "label": int(preds[name])}

    base = request.url_root.rstrip("/")
    if base.startswith("http://"):
        base = "https://" + base[len("http://"):]
    result["_video_url"] = f"{base}/video/{overlay_name}"

    result["_frame_count"] = int(frame_count)
    result["_fps"] = float(fps)
    result["_metrics"] = metrics

    return jsonify(result)


@app.route("/video/<path:filename>", methods=["GET"])
@cross_origin()
def get_video(filename):
    """Serve processed overlay video files."""
    return send_from_directory(OVERLAY_DIR, filename, mimetype="video/webm")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
