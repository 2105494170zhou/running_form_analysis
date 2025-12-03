from pathlib import Path
import os

from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename

# Import your inference helper and label names
from inference_running_functions import run_inference_on_video, LABEL_NAMES

# -----------------------------------------------------------------------------
# App setup
# -----------------------------------------------------------------------------
app = Flask(__name__)

# Global CORS config: allow ALL origins on ALL routes (no credentials)
CORS(
    app,
    resources={r"/*": {"origins": "*"}},
    supports_credentials=False,
)

# Temp directory for uploaded videos
UPLOAD_DIR = Path("/tmp/videos")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.route("/health", methods=["GET"])
@cross_origin()  # make sure CORS headers are applied
def health():
    """
    Simple health check endpoint.
    """
    return jsonify({"status": "ok"})


@app.route("/", methods=["GET"])
@cross_origin()
def index():
    """
    Root endpoint so hitting the bare URL returns something useful.
    """
    return jsonify(
        {
            "message": "Running Form API is alive. Use POST /analyze with a 'video' file."
        }
    )


@app.route("/analyze", methods=["POST"])
@cross_origin()
def analyze():
    """
    Main endpoint used by Framer.

    Expects:
      - POST /analyze
      - Content-Type: multipart/form-data
      - A single file field named 'video' containing the uploaded video.

    Returns:
      - On success: JSON of probabilities and labels for each running-form issue.
      - On error: JSON with 'error' and appropriate HTTP status code.
    """
    # 1. Check the 'video' file is present
    if "video" not in request.files:
        return jsonify({"error": "No file field named 'video' in the request."}), 400

    file = request.files["video"]

    # 2. Basic validation of filename
    if not file.filename:
        return jsonify({"error": "Uploaded file has an empty filename."}), 400

    # 3. Sanitize filename and save to /tmp/videos
    filename = secure_filename(file.filename)
    save_path = UPLOAD_DIR / filename
    file.save(save_path)

    try:
        # 4. Run your existing inference pipeline
        probs, preds = run_inference_on_video(save_path)
    except Exception as e:
        # Log to server stdout for debugging on Render
        print(f"[ERROR] Inference failed for {save_path}: {e}", flush=True)
        return jsonify({"error": f"Inference failed: {str(e)}"}), 500
    finally:
        # 5. Clean up temporary file
        try:
            os.remove(save_path)
        except OSError:
            pass

    # 6. Build JSON result in a Framer-friendly format
    result = {
        name: {
            "prob": float(probs[name]),
            "label": int(preds[name]),
        }
        for name in LABEL_NAMES
    }

    return jsonify(result)


# -----------------------------------------------------------------------------
# Local dev entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # For local testing:
    #   python webrun.py
    # Then open http://127.0.0.1:5000/health in your browser.
    app.run(host="0.0.0.0", port=5000, debug=True)
