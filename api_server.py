# api_server.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import tensorflow as tf
import json
import base64
import os
import sys

# ============================================================
# FLASK APP SETUP
# ============================================================
app = Flask(__name__)
CORS(app)

# ============================================================
# PATHS (CORRECTED TO MATCH YOUR STRUCTURE)
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs_final_fixed")

BINARY_MODEL_PATH = os.path.join(
    OUTPUT_DIR, "best_plastic_binary_model.keras"
)
TYPE_MODEL_PATH = os.path.join(
    OUTPUT_DIR, "best_model.h5"
)
LABEL_PATH = os.path.join(
    OUTPUT_DIR, "label_encoder.json"
)

# ============================================================
# CONFIG
# ============================================================
BIN_SIZE = 160
TYPE_SIZE = 224

# ============================================================
# GLOBALS
# ============================================================
binary_model = None
type_model = None
class_labels = []

# ============================================================
# MICRON MAP
# ============================================================
MICRON_MAP = {
    "ABS": "100–150 µm", "EVA": "40–100 µm", "HDPE": "60–120 µm",
    "LDPE": "30–80 µm", "PA": "70–130 µm", "PBT": "80–150 µm",
    "PC": "100–200 µm", "PEEK": "50–120 µm", "PET": "70–120 µm",
    "PMMA": "80–150 µm", "PP": "60–110 µm", "PS": "80–120 µm",
    "PTFE": "100–200 µm", "PVC": "80–150 µm", "SAN": "90–140 µm",
    "TPU": "50–100 µm", "PE": "30–200 µm"
}

# ============================================================
# LOAD MODELS & LABELS (FIXED)
# ============================================================
def load_models():
    global binary_model, type_model, class_labels

    print("📁 BASE_DIR:", BASE_DIR)
    print("📁 OUTPUT_DIR:", OUTPUT_DIR)

    if not os.path.exists(OUTPUT_DIR):
        print("❌ outputs_final_fixed folder NOT FOUND")
        sys.exit(1)

    print("📂 Files found:")
    for f in os.listdir(OUTPUT_DIR):
        print("  -", f)

    # ------------------ Binary Model ------------------
    print("🔄 Loading Binary Model...")
    if not os.path.exists(BINARY_MODEL_PATH):
        print("❌ Binary model NOT FOUND:", BINARY_MODEL_PATH)
        sys.exit(1)

    binary_model = tf.keras.models.load_model(BINARY_MODEL_PATH)
    print("✅ Binary model loaded")

    # ------------------ Type Model ------------------
    print("🔄 Loading Type Model...")
    if not os.path.exists(TYPE_MODEL_PATH):
        print("❌ Type model NOT FOUND:", TYPE_MODEL_PATH)
        sys.exit(1)

    type_model = tf.keras.models.load_model(TYPE_MODEL_PATH)
    print("✅ Type model loaded")

    # ------------------ Labels (FIXED FOR LabelEncoder) ------------------
    print("📄 Loading Labels...")
    if not os.path.exists(LABEL_PATH):
        print("❌ Label file NOT FOUND:", LABEL_PATH)
        sys.exit(1)

    with open(LABEL_PATH, "r") as f:
        label_map = json.load(f)

    # label_encoder.json format: { "ABS": 0, "EVA": 1, ... }
    class_labels = [None] * len(label_map)
    for class_name, idx in label_map.items():
        class_labels[int(idx)] = class_name

    print("✅ Labels loaded:", class_labels)

# ============================================================
# IMAGE HELPERS
# ============================================================
def preprocess_image(img, size):
    img = cv2.resize(img, (size, size))
    img = img.astype("float32")
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return np.expand_dims(img, axis=0)

def encode_image_to_dataurl(img):
    _, buffer = cv2.imencode(".jpg", img)
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode()

# ============================================================
# PREDICTION LOGIC
# ============================================================
def predict_image(img):
    # ---------- Binary ----------
    x_bin = preprocess_image(img, BIN_SIZE)
    bin_pred = binary_model.predict(x_bin, verbose=0)

    plastic_prob = float(bin_pred[0][0])
    non_plastic = 1 - plastic_prob

    if plastic_prob < 0.5:
        return {
            "label": "Non-Plastic",
            "plastic": round(plastic_prob, 4),
            "non_plastic": round(non_plastic, 4)
        }

    # ---------- Type ----------
    x_type = preprocess_image(img, TYPE_SIZE)
    probs = type_model.predict(x_type, verbose=0)[0]
    idx = int(np.argmax(probs))
    plastic_type = class_labels[idx]

    return {
        "label": "Plastic",
        "plastic_type": plastic_type,
        "confidence": round(float(probs[idx]), 4),
        "micron": MICRON_MAP.get(plastic_type, "N/A")
    }

# ============================================================
# ROUTES
# ============================================================
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "binary_loaded": binary_model is not None,
        "type_loaded": type_model is not None,
        "classes": class_labels
    })

@app.route("/api/predict_file", methods=["POST"])
def predict_file():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    img = cv2.imdecode(
        np.frombuffer(file.read(), np.uint8),
        cv2.IMREAD_COLOR
    )

    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    result = predict_image(img)
    result["image"] = encode_image_to_dataurl(img)
    return jsonify(result)

# ============================================================
# MAIN (CRITICAL FIX: NO RELOADER)
# ============================================================
if __name__ == "__main__":
    print("🚀 Starting Plastic Detection API...")
    load_models()

    print("🌐 API running at http://127.0.0.1:5001")
    app.run(
        host="0.0.0.0",
        port=5001,
        debug=False,        # IMPORTANT
        use_reloader=False # CRITICAL FOR WINDOWS + TF
    )
