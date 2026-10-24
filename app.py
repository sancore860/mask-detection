"""
Face Mask Monitoring System - Flask API
---------------------------------------
Endpoints
  GET  /          -> live monitoring web page (uses the browser webcam)
  GET  /health    -> health check
  POST /predict   -> send an image (form field "image"), get JSON with faces + mask status
                     add ?annotate=1 to also receive the image with boxes drawn (base64)
"""
import os
import base64

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from tensorflow.keras.models import load_model

# ---------------- Config (must match your training notebook) ----------------
IMG_SIZE = 128                                   # same as IMG_SIZE in notebook
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_simple_cnn_mask.h5")
# train_gen.class_indices = {'with_mask': 0, 'without_mask': 1}
# sigmoid output  > 0.5  ->  without_mask,  <= 0.5 -> with_mask
THRESHOLD = 0.5
FACE_PADDING = 0.15                              # extra margin around detected face

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024   # 5 MB upload limit

# ---------------- Load model & face detector once at startup ----------------
print(f"[INFO] Loading model from {MODEL_PATH} ...")
model = load_model(MODEL_PATH, compile=False)
print("[INFO] Model loaded.")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
if face_cascade.empty():
    raise RuntimeError("Could not load Haar cascade for face detection.")


# ---------------- Helpers ----------------
def read_image_from_request():
    """Return a BGR image (numpy array) from the uploaded file, or None."""
    file = request.files.get("image")
    if file is None:
        return None
    data = np.frombuffer(file.read(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def detect_faces(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=3, minSize=(40, 40)
    )
    return [tuple(int(v) for v in f) for f in faces]


def preprocess_face(img_bgr, box):
    """Crop face (with padding) and prepare it exactly like training."""
    h, w = img_bgr.shape[:2]
    x, y, bw, bh = box
    px, py = int(bw * FACE_PADDING), int(bh * FACE_PADDING)
    x1, y1 = max(0, x - px), max(0, y - py)
    x2, y2 = min(w, x + bw + px), min(h, y + bh + py)
    face = img_bgr[y1:y2, x1:x2]
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)      # Keras trained on RGB
    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
    return face.astype("float32") / 255.0              # rescale=1./255


def classify_faces(img_bgr, boxes):
    batch = np.stack([preprocess_face(img_bgr, b) for b in boxes])
    probs = model.predict(batch, verbose=0).flatten()
    results = []
    for (x, y, w, h), p in zip(boxes, probs):
        no_mask = float(p) > THRESHOLD
        confidence = float(p) if no_mask else 1.0 - float(p)
        results.append({
            "box": {"x": x, "y": y, "w": w, "h": h},
            "label": "No Mask" if no_mask else "Mask",
            "with_mask": not no_mask,
            "confidence": round(confidence, 4),
        })
    return results


def draw_annotations(img_bgr, results):
    out = img_bgr.copy()
    for r in results:
        b = r["box"]
        color = (0, 200, 0) if r["with_mask"] else (0, 0, 255)   # BGR
        cv2.rectangle(out, (b["x"], b["y"]), (b["x"] + b["w"], b["y"] + b["h"]), color, 2)
        text = f'{r["label"]} {r["confidence"] * 100:.0f}%'
        cv2.putText(out, text, (b["x"], max(20, b["y"] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out


# ---------------- Routes ----------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify(status="ok", model=os.path.basename(MODEL_PATH))


@app.route("/predict", methods=["POST"])
def predict():
    img = read_image_from_request()
    if img is None:
        return jsonify(error="Send an image file in form field 'image'."), 400

    boxes = detect_faces(img)
    results = classify_faces(img, boxes) if boxes else []

    response = {
        "image_width": img.shape[1],
        "image_height": img.shape[0],
        "faces_detected": len(results),
        "with_mask": sum(r["with_mask"] for r in results),
        "without_mask": sum(not r["with_mask"] for r in results),
        "faces": results,
    }

    if request.args.get("annotate") == "1":
        ok, buf = cv2.imencode(".jpg", draw_annotations(img, results))
        if ok:
            response["annotated_image"] = (
                "data:image/jpeg;base64," + base64.b64encode(buf).decode()
            )
    return jsonify(response)


@app.errorhandler(413)
def too_large(_):
    return jsonify(error="Image too large (max 5 MB)."), 413


if __name__ == "__main__":
    # Development server only. For production use gunicorn (see README steps).
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)