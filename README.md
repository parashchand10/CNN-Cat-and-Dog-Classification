# PetVision AI — Cat vs. Dog Classifier

A lightweight Flask web app that classifies uploaded images as **cat** or **dog** in real time, powered by a TensorFlow Lite model running through `ai-edge-litert`.

## Demo

| Input | Result |
|---|---|
| ![Input screen](petvision-ai-input-screen.png) | ![Classification result](petvision-ai-classification-result.png) |

## Features

- Upload any image (PNG, JPG, HEIC, etc.) and get an instant cat/dog prediction
- Confidence percentages for both classes, with a "too close to call" fallback when scores are near 50/50
- Runs on the lightweight `.tflite` model — no full TensorFlow install required
- Simple, dependency-light Flask backend, ready to deploy on Render

## Tech Stack

- **Backend:** Flask, Gunicorn
- **Inference:** TensorFlow Lite via `ai-edge-litert`
- **Image processing:** Pillow, NumPy
- **Frontend:** HTML/CSS (Jinja templates)

## Project Structure

```
cat-dog-classifier/
├── app.py                  # Flask app + TFLite inference
├── requirements.txt
├── Procfile                 # Render / Heroku start command
├── assets/                  # README screenshots
├── model/
│   └── cat_dog_cnn_model.tflite
├── static/
│   ├── dog.png              # background image
│   └── uploads/             # saved user uploads (preview)
└── templates/
    └── index.html
```

## Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
git clone https://github.com/<your-username>/cat-dog-classifier.git
cd cat-dog-classifier
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Run Locally

```bash
python app.py
```

Then open `http://localhost:5000` and upload an image.

## How It Works

1. An uploaded image is resized to `150×150` to match the model's training input.
2. The raw pixel array is passed to the TFLite interpreter (no `/255` rescaling — normalization is baked into the model graph).
3. The model outputs a single sigmoid score: closer to `1` → dog, closer to `0` → cat.
4. Scores within a 10-point band of 50/50 are reported as "too close to call."

## Deployment (Render)

1. Push this repository to GitHub.
2. On [Render](https://render.com): **New → Web Service** → connect the repo.
3. **Build command:** `pip install -r requirements.txt`
4. **Start command:** `gunicorn app:app`

> **Note:** Render's free tier uses an ephemeral filesystem, so files saved to `static/uploads/` won't persist across restarts.
