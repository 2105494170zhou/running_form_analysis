from pathlib import Path
import os

from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename

# Import the helper and label names from your inference module
from inference_running_functions import run_inference_on_video, LABEL_NAMES

# 1. Create the Flask app
app = Flask(__name__)

# 2. Enable CORS on ALL routes, for ALL origins (no credentials)
#    This will let requests from framer.com, framercanvas.com, framer.app, etc.
CORS(
    app,
    resources={r"/*": {"origins": "*"}},
    supports_credentials=False,
)

# 3. Directory where we'll temporarily save uploaded videos
UPLOAD_DIR = Path("/tmp/videos")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------
# Health-check endpoint
# -------------------------
@app.route("/health", methods=["GET"])
@cross_origin(origins="*", send_wildcard=True)
def health():
    """
    Health-check endpoint.

    Use this to verify that:
      - The Flask app is running.
      - CORS headers are being sent.
    """
    return jsonify({"status": "ok"})


# -------------------------
# Root endpoint (optional)
# -------------------------
@app.route("/", methods=["GET"])
@cross_origin(origins="*", send_wildcard=True)
def index():
    """
    Simple index route so visiting the bare URL doesn't show a 404.

    Example:
      https://running-form-analysis.onrender.com/
    """
    return jsonify(
        {"message": "Running Form API is alive. Use POST /analyze to analyze a video."}
    )


# -------------------------
# Main inference endpoint
# -------------------------
@app.route("/analyze", methods=["POST"])
@cross_origin(origins="*", send_wildcard=True)
def analyze():
    """
    Main endpoint used by Framer / clients.

    Expected request:
      - HTTP method: POST
      - Content-Type: multipart/form-data
      - A single file field named 'video' containing the uploaded video.

    Returns:
      - On success: JSON like:
          {
            "overstriding":      {"prob": 0.83, "label": 1},
            "hard_landing":      {"prob": 0.12, "label": 0},
            "forward_lean":      {"prob": 0.45, "label": 0},
            "little_propulsion": {"prob": 0.67, "label": 1}
          }
      - On error: JSON with an "error" message and HTTP status 4xx/5xx.
    """
    # 1. Check that the 'video' file is present in the form
    if "video" not in request.files:
        return jsonify({"error": "No file field named 'video' in the request."}), 400

    file = request.files["video"]

    # 2. Validate the filename
    if file.filename == "":
        return jsonify({"error": "Uploaded file has an empty filename."}), 400

    # 3. Sanitize the filename and save to /tmp
    filename = secure_filename(file.filename)
    save_path = UPLOAD_DIR / filename
    file.save(save_path)

    try:
        # 4. Run your existing inference pipeline on the saved file
        probs, preds = run_inference_on_video(save_path)
    except Exception as e:
        # Return 500 with error message if inference fails
        return jsonify({"error": f"Inference failed: {str(e)}"}), 500
    finally:
        # 5. Clean up: remove the video file to save space
        try:
            os.remove(save_path)
        except OSError:
            pass

    # 6. Build JSON result
    result = {}
    for name in LABEL_NAMES:
        result[name] = {
            "prob": float(probs[name]),
            "label": int(preds[name]),
        }

    return jsonify(result)


# -------------------------
# Local dev entry point
# -------------------------
if __name__ == "__main__":
    # Run locally with:
    #   python webrun.py
    # then visit http://127.0.0.1:5000/health
    app.run(host="0.0.0.0", port=5000, debug=True)
