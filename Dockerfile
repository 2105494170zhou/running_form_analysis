# Use a slim Python image
FROM python:3.11-slim

# Create working directory
WORKDIR /app

# (Optional but recommended) Install system dependencies for video processing
# Remove or adjust if you don't need ffmpeg or other tools.
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# Make sure requirements.txt includes:
#   flask
#   flask-cors
#   (and whatever you use in inference_running_functions)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app (including webrun.py and model files)
COPY . .

# Hugging Face (and many PaaS) expect the app to listen on this port
ENV PORT=7860
ENV PYTHONUNBUFFERED=1

# Expose the port
EXPOSE 7860

# Start the Flask app via gunicorn.
# - "webrun:app" means: use `app` object from webrun.py
CMD ["gunicorn", "webrun:app", "--bind", "0.0.0.0:7860", "--workers", "1"]
