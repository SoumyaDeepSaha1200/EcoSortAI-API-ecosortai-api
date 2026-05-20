from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import tensorflow as tf
import json
import base64
import os
import sys
import traceback

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs_final_fixed")

BINARY_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_plastic_binary_model.keras")
TYPE_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.h5")
LABEL_PATH = os.path.join(OUTPUT_DIR, "label_encoder.json")

BIN_SIZE = 160
TYPE_SIZE = 224

binary_model = None
type_model = None
class_labels = []

MICRON_MAP = {
    "ABS": "100–150 µm", "EVA": "40–100 µm", "HDPE": "60–120 µm",
    "LDPE": "30–80 µm", "PA": "70–130 µm", "PBT": "80–150 µm",
    "PC": "100–200 µm", "PEEK": "50–120 µm", "PET": "70–120 µm",
    "PMMA": "80–150 µm", "PP": "60–110 µm", "PS": "80–120 µm",
    "PTFE": "100–200 µm", "PVC": "80–150 µm", "SAN": "90–140 µm",
    "TPU": "50–100 µm", "PE": "30–200 µm"
}


def load_models():
    global binary_model, type_model, class_labels

    print("BASE_DIR:", BASE_DIR)
    print("OUTPUT_DIR:", OUTPUT_DIR)

    if not os.path.exists(OUTPUT_DIR):
        print("outputs_final_fixed folder not found")
        sys.exit(1)

    if not os.path.exists(BINARY_MODEL_PATH):
        print("Binary model not found:", BINARY_MODEL_PATH)
        sys.exit(1)

    if not os.path.exists(TYPE_MODEL_PATH):
        print("Type model not found:", TYPE_MODEL_PATH)
        sys.exit(1)

    if not os.path.exists(LABEL_PATH):
        print("Label file not found:", LABEL_PATH)
        sys.exit(1)

    print("Loading binary model...")
    binary_model = tf.keras.models.load_model(BINARY_MODEL_PATH)
    print("Binary model loaded")

    print("Loading type model...")
    type_model = tf.keras.models.load_model(TYPE_MODEL_PATH)
    print("Type model loaded")

    with open(LABEL_PATH, "r") as f:
        label_map = json.load(f)

    class_labels = [None] * len(label_map)

    for class_name, idx in label_map.items():
        class_labels[int(idx)] = class_name

    print("Labels loaded:", class_labels)


def preprocess_image(img, size):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size))
    img = img.astype("float32")
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return np.expand_dims(img, axis=0)


def encode_image_to_dataurl(img):
    ok, buffer = cv2.imencode(".jpg", img)

    if not ok:
        return None

    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode()


def predict_image(img):
    if binary_model is None or type_model is None:
        raise Exception("Models are not loaded")

    x_bin = preprocess_image(img, BIN_SIZE)
    bin_pred = binary_model.predict(x_bin, verbose=0)

    plastic_prob = float(bin_pred[0][0])
    non_plastic_prob = 1.0 - plastic_prob

    if plastic_prob < 0.5:
        return {
            "label": "Non-Plastic",
            "plastic": round(plastic_prob, 4),
            "non_plastic": round(non_plastic_prob, 4),
            "confidence": round(non_plastic_prob, 4)
        }

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


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": "Internal server error",
        "details": str(error)
    }), 500


@app.route("/api/test", methods=["GET"])
def test():
    return jsonify({"message": "API working"})


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "binary_loaded": binary_model is not None,
        "type_loaded": type_model is not None,
        "classes": class_labels
    })


@app.route("/api/debug_predict", methods=["POST"])
def debug_predict():
    try:
        file = request.files.get("file")

        if not file:
            return jsonify({"step": "file_check", "error": "No file uploaded"}), 400

        image_bytes = file.read()

        img = cv2.imdecode(
            np.frombuffer(image_bytes, np.uint8),
            cv2.IMREAD_COLOR
        )

        if img is None:
            return jsonify({"step": "decode", "error": "Invalid image"}), 400

        return jsonify({
            "message": "Image received and decoded successfully",
            "filename": file.filename,
            "image_shape": img.shape
        })

    except Exception as e:
        return jsonify({
            "step": "exception",
            "error": str(e),
            "trace": traceback.format_exc()
        }), 500


@app.route("/api/predict_file", methods=["POST"])
def predict_file():
    try:
        file = request.files.get("file")

        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        image_bytes = file.read()

        img = cv2.imdecode(
            np.frombuffer(image_bytes, np.uint8),
            cv2.IMREAD_COLOR
        )

        if img is None:
            return jsonify({"error": "Invalid image"}), 400

        result = predict_image(img)

        result["image"] = encode_image_to_dataurl(img)

        return jsonify(result)

    except Exception as e:
        return jsonify({
            "error": "Prediction failed",
            "details": str(e),
            "trace": traceback.format_exc()
        }), 500


print("Loading models...")
load_models()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        use_reloader=False
    )