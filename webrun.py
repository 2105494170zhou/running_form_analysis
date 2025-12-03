from pathlib import Path
import os

from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename

# Import the helper and label names from your inference module
from inference_running_functions import run_inference_on_video, LABEL_NAMES

# 1. Create the Flask app
app = Flask(__name__)

# Allow all origins on all routes
CORS(app, resources={r"/*": {"origins": "*"}})


# 3. Directory where we'll temporarily save uploaded videos
#    /tmp is a standard writable temp folder on Linux (Render uses Linux containers)
UPLOAD_DIR = Path("/tmp/videos")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------
# Health-check endpoint
# -------------------------
@app.route("/health", methods=["GET"])
@cross_origin()  # explicitly add CORS headers on this route
def health():
    """
    Health-check endpoint.

    Use this to verify that:
      - The Flask app is running.
      - The route is registered correctly.
    """
    return jsonify({"status": "ok"})


# -------------------------
# Root endpoint (optional)
# -------------------------
@app.route("/", methods=["GET"])
def index():
    """
    Simple index route so visiting the bare URL doesn't show a 404.

    Example:
      http://127.0.0.1:5000/
    """
    return jsonify({"message": "Running Form API is alive. Use POST /analyze to analyze a video."})


# -------------------------
# Main inference endpoint
# -------------------------
@app.route("/analyze", methods=["POST"])
@cross_origin()  # explicitly add CORS headers on this route
def analyze():
    """
    Main endpoint used by Framer / clients.

    Expected request:
      - HTTP method: POST
      - Content-Type: multipart/form-data
      - A single file field named 'video' containing the uploaded video.

    Behavior:
      - Saves the uploaded video to /tmp/videos.
      - Calls run_inference_on_video() from your inference module.
      - Builds a JSON object with probabilities and predicted labels.

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

    # 3. Sanitize the filename (removes dangerous characters, relative paths, etc.)
    filename = secure_filename(file.filename)

    # 4. Save the file to the temporary directory
    save_path = UPLOAD_DIR / filename
    file.save(save_path)

    try:
        # 5. Run your existing inference pipeline on the saved file
        probs, preds = run_inference_on_video(save_path)
    except Exception as e:
        # If anything goes wrong in the pipeline, return an error to the client.
        # In a real deployment you'd also log this for debugging.
        return jsonify({"error": f"Inference failed: {str(e)}"}), 500
    finally:
        # 6. Clean up: remove the video file to save space.
        try:
            os.remove(save_path)
        except OSError:
            # If deletion fails, we just ignore the error.
            pass

    # 7. Build a dictionary to return as JSON
    result = {}
    for name in LABEL_NAMES:
        result[name] = {
            "prob": float(probs[name]),   # ensure it's a plain Python float
            "label": int(preds[name]),    # ensure it's a plain Python int (0 or 1)
        }

    # 8. Send the JSON response back to the client (Framer frontend)
    return jsonify(result)


# -------------------------
# Local dev entry point
# -------------------------
if __name__ == "__main__":
    # This allows you to run the Flask app locally for testing:
    #   python app.py
    # Visit http://127.0.0.1:5000/health to test.
    app.run(host="0.0.0.0", port=5000, debug=True)
