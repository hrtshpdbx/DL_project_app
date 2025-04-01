from flask import Flask, request, jsonify, send_file, send_from_directory, make_response
from flask_cors import CORS
import tensorflow as tf
from pymongo import MongoClient
from pymongo.server_api import ServerApi
import gridfs
from werkzeug.utils import secure_filename
import os
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import time
import shutil
from dotenv import load_dotenv
from utils import *

#CHANGE STARTS HERE
import os

# Prevent TensorFlow from registering CUDA libraries multiple times
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# Disable OneDNN optimizations if needed
#os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file

# Check if variables are loaded correctly
print("MongoDB URI:", os.getenv("MONGO_URI"))  # Debugging

#CHANGE ENDS HERE

# Flask Setup
app = Flask(__name__)
#INSTEAD OF THIS: CORS(app)  # Allows requests from any frontend
CORS(app, resources={r"/*": {"origins": os.getenv("CORS_ORIGINS", "*")}}) #THIS

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# MongoDB Connection
load_dotenv(".env")
URI = os.getenv("MONGO_URI")
client = MongoClient(URI, server_api=ServerApi('1'))

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

db = client["IT353Project"]
fs = gridfs.GridFS(db)
model = None  # Global model variable


def load_model_by_dataset(dataset_name):
    """Fetches and loads the model corresponding to the given dataset name."""
    file_data = fs.find_one({"metadata.dataset": dataset_name})
    
    if file_data:
        temp_model_path = "temp_model.keras"
        with open(temp_model_path, "wb") as temp_file:
            temp_file.write(file_data.read())

        model = tf.keras.models.load_model(temp_model_path)
        return model
    else:
        return None


@app.route("/load_model", methods=["POST"])
def load_model():
    """Loads the model based on dataset name."""
    data = request.json
    dataset_name = data.get("dataset_name")

    global model
    model = load_model_by_dataset(dataset_name)

    if model:
        return jsonify({"message": f"Model for {dataset_name} loaded successfully!"})
    else:
        return jsonify({"error": "Model not found"}), 404


@app.route("/predict", methods=["POST"])
def predict():
    """Handles image upload, inference, and visualization generation."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    # Clear uploads folder
    shutil.rmtree(app.config["UPLOAD_FOLDER"])
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    file = request.files["file"]
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    # Preprocess Image
    img = Image.open(file_path).resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Perform Inference
    if model is None:
        return jsonify({"error": "Model not loaded"}), 400

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]

    # Generate BatchNorm Visualization
    before_bn, after_bn = visualize_batch_norm_effect(model, "batch_normalization_4", img_array[0])

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(before_bn, bins=50, alpha=0.7, label="Before BN")
    plt.title("Activation Distribution Before BN")

    plt.subplot(1, 2, 2)
    plt.hist(after_bn, bins=50, alpha=0.7, label="After BN", color="orange")
    plt.title("Activation Distribution After BN")

    batch_norm_image_path = os.path.join(app.config["UPLOAD_FOLDER"], "batch_norm_vis.png")
    plt.savefig(batch_norm_image_path)
    plt.close()
    print("✅ BatchNorm image saved at:", batch_norm_image_path, os.path.exists(batch_norm_image_path))  # Debug

    # Generate Grad-CAM Image
    grad_cam_image = grad_cam_heatmap(model, img_array[0], "multiply_4", predicted_class)
    grad_cam_image_path = os.path.join(app.config["UPLOAD_FOLDER"], f"gradcam_vis.png")
    overlay_img = overlay_heatmap(img_array[0], grad_cam_image)
    Image.fromarray((overlay_img * 255).astype(np.uint8)).save(grad_cam_image_path)
    print("✅ Grad-CAM image saved at:", grad_cam_image_path, os.path.exists(grad_cam_image_path))  # Debug

    timestamp = int(time.time())  # Unique value for cache busting
    base_url = request.host_url.rstrip("/")  # Get the correct base URL

    return jsonify({
        "predicted_class": int(predicted_class),
        "batch_norm_vis_url": f"{base_url}/uploads/batch_norm_vis.png?t={timestamp}",
        "grad_cam_vis_url": f"{base_url}/uploads/gradcam_vis.png?t={timestamp}"
    })


@app.route('/uploads/<path:filename>')
def serve_uploads(filename):
    """Serve static files from the uploads directory."""
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/download/<image_type>")
def download_image(image_type):
    """Allows frontend to download batch norm or gradcam image."""
    if image_type == "batch_norm":
        path = os.path.join(app.config["UPLOAD_FOLDER"], "batch_norm_vis.png")
    elif image_type == "gradcam":
        path = os.path.join(app.config["UPLOAD_FOLDER"], "gradcam_vis.png")
    else:
        return jsonify({"error": "Invalid image type"}), 400

    # Create response and explicitly allow CORS
    response = make_response(send_file(path, mimetype="image/png"))
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

'''INSTEAD OF THIS
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use PORT from environment, default to 5000
    app.run(host="0.0.0.0", port=port, debug=True)
'''

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")  # Bind to all network interfaces
    port = int(os.getenv("PORT", 5000))
    app.run(host=host, port=port, debug=True)
