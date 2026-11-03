# 😷 Face Mask Detection

A deep learning web app that detects whether a person in an uploaded photo is **wearing a face mask or not**. A custom CNN built with TensorFlow/Keras does the classification, and a Flask backend with an HTML/CSS front end serves it.

**🔗 Live demo:** [https://YOUR-SERVICE-NAME.onrender.com](https://mask-detection-79q2.onrender.com/)

> The app is hosted on Render's free plan, so it sleeps after 15 minutes without visitors. The first load after a quiet period can take about a minute.

---

## Features

- Upload a face photo by click or drag and drop
- Instant prediction: **With Mask** or **Without Mask**
- Confidence score shown with a progress bar
- Simple REST endpoint (`/predict`) that can be reused by other apps
- Deployable with Docker (Hugging Face Spaces) or as a Python service (Render)

---

## Dataset

| Item | Value |
|---|---|
| Folder | `face-mask-dataset/` |
| Classes | `with_mask` (3,725 images), `without_mask` (3,828 images) |
| Total images | 7,553 (roughly balanced) |
| Training set | 6,043 images (80%) |
| Validation set | 1,510 images (20%) |
| Label mapping | `with_mask = 0`, `without_mask = 1` |

The dataset is not included in this repository because of its size.

---

## Data Preparation

- Images are resized to **128 × 128** pixels with 3 colour channels
- Pixel values are scaled to the range 0–1 (`rescale=1/255`)
- Labels are binary (`class_mode="binary"`)
- 80/20 train/validation split with a fixed seed (`42`)
- **Data augmentation** (via `ImageDataGenerator`):
  - rotation up to 20°
  - zoom range 0.15
  - width and height shift 0.2
  - shear range 0.15
  - horizontal flip
  - brightness range 0.8 – 1.2
  - `fill_mode="nearest"`

---

## Model Architecture

A Sequential CNN with four convolution blocks followed by a classifier head.

| Layer | Output shape |
|---|---|
| Conv2D (32, 3×3, ReLU) + BatchNorm + MaxPool | 64 × 64 × 32 |
| Conv2D (64, 3×3, ReLU) + BatchNorm + MaxPool | 32 × 32 × 64 |
| Conv2D (128, 3×3, ReLU) + BatchNorm + MaxPool | 16 × 16 × 128 |
| Conv2D (256, 3×3, ReLU) + BatchNorm + MaxPool | 8 × 8 × 256 |
| Flatten | 16,384 |
| Dense (256, ReLU) | 256 |
| Dropout (0.5) | 256 |
| Dense (1, Sigmoid) | 1 |

**Total parameters:** 4,585,153 (about 17.5 MB)

The output is the probability of `without_mask`. A value above 0.5 is classified as **Without Mask**.

---

## Training Setup

| Setting | Value |
|---|---|
| Optimizer | Adam (learning rate 1e-3) |
| Loss | Binary cross-entropy |
| Metrics | Accuracy, Precision, Recall |
| Batch size | 32 |
| Max epochs | 30 |
| Callbacks | `EarlyStopping` (val_loss, patience 6), `ReduceLROnPlateau` (factor 0.5, patience 3), `ModelCheckpoint` (best val_accuracy) |

The best model is saved as `best_simple_cnn_mask.h5`.

---

## Results

| Metric | Value |
|---|---|
| Validation accuracy | **98.01%** |
| Precision | 98.93% |
| Recall | 97.12% |

**Confusion matrix** (validation set):

|  | Predicted: with_mask | Predicted: without_mask |
|---|---|---|
| **Actual: with_mask** | 736 | 9 |
| **Actual: without_mask** | 20 | 745 |

---

## Project Structure

```
mask-detection/
├── app.py                        # Flask backend
├── best_simple_cnn_mask.h5       # Trained model
├── mask_detection.ipynb          # Training notebook
├── requirements.txt              # Python dependencies
├── Dockerfile                    # For Docker / Hugging Face Spaces
├── templates/
│   └── index.html                # Web page
└── static/
    └── style.css                 # Styling
```

---

## Run Locally

**Requirements:** Python 3.10 – 3.12 and TensorFlow.

```bash
# 1. Clone the repository
git clone https://github.com/sancore860/mask-detection.git
cd mask-detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the app
python app.py
```

Open **http://127.0.0.1:5000** in your browser, upload a face photo and click **Detect**.

---

## API

**`POST /predict`**

Send an image as form data with the field name `image`.

```bash
curl -X POST -F "image=@photo.jpg" http://127.0.0.1:5000/predict
```

Response:

```json
{
  "label": "With Mask",
  "confidence": 98.7,
  "with_mask": true
}
```

**`GET /health`** returns `{"status": "ok"}`.

---

## Deployment

GitHub only stores the code and cannot run a Flask app. Use one of these hosts, both of which deploy from this repository:

- **Hugging Face Spaces** (Docker): create a Space with the Docker SDK, upload the files, and set `app_port: 7860` in the Space's README header. Free hardware offers 16 GB of RAM, but check whether your account can create Docker Spaces on the free tier.
- **Render** (Python web service):
  - Build command: `pip install -r requirements.txt`
  - Start command: `gunicorn app:app --workers 1 --threads 2 --timeout 120`
  - Environment variable: `PYTHON_VERSION = 3.11.9`
  - The free plan has 512 MB of RAM, which may not be enough for TensorFlow.

---

## Known Limitations

- The model was trained **only on face photos**. It has no "no face" class, so an image with no face (for example a wall) may be wrongly labelled as **Without Mask**.
- It works best on **close-up, front-facing photos of one face**. Group photos and full-body shots give unreliable results.
- The validation set used the same augmentation as training, so reported numbers may differ slightly on clean images.

### Planned improvements

- Add face detection (OpenCV) to crop each face before classification and show "No face detected" when none is found
- Add a `no_face` class with non-face images and retrain
- Use a clean, non-augmented validation set
- Support multiple faces in one image

---

## Tech Stack

Python · TensorFlow / Keras · Flask · Gunicorn · Pillow · NumPy · HTML · CSS · JavaScript · Docker

---

## Author

**sancore860** — [GitHub](https://github.com/sancore860)
