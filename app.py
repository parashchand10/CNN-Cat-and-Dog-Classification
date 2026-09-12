import os
import uuid

import numpy as np
from flask import Flask, render_template, request
from PIL import Image

# ai-edge-litert is the maintained successor to tf.lite.Interpreter for
# running .tflite models outside of a full TensorFlow install.
from ai_edge_litert.interpreter import Interpreter

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "cat_dog_cnn_model.tflite")
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
IMG_SIZE = (150, 150)  # must match training (see Fine_Tune.ipynb)

# "Too close to call" band around 50/50 -> report both instead of one winner.
TIE_BAND = 10.0  # percentage points

os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load the model once at startup, not per-request.
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def preprocess_image(img: Image.Image) -> np.ndarray:
    """Resize + format an uploaded image to match the model's expected input.

    Note: the notebook's Keras model has `preprocess_input` (MobileNetV2
    normalization) baked in as a layer, so we only resize here and pass
    raw 0-255 float values, exactly like the notebook's own inference cell.
    """
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    return arr


def run_inference(img: Image.Image) -> dict:
    arr = preprocess_image(img)
    interpreter.set_tensor(input_details[0]["index"], arr)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])
    score = float(output[0][0])  # sigmoid output: closer to 1 -> dog, closer to 0 -> cat

    dog_pct = score * 100
    cat_pct = (1 - score) * 100

    if abs(dog_pct - cat_pct) <= TIE_BAND:
        winner = "both"
    elif dog_pct > cat_pct:
        winner = "dog"
    else:
        winner = "cat"

    return {"cat_pct": cat_pct, "dog_pct": dog_pct, "winner": winner}


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    img_path = None
    error_message = None

    if request.method == "POST":
        file = request.files.get("file")

        if not file or file.filename == "":
            error_message = "Please choose an image file."
        else:
            try:
                # Let Pillow decide if it's a real image, instead of trusting
                # the filename's extension (fixes .jfif/.heic/etc. being
                # wrongly rejected even though they're valid images).
                img = Image.open(file.stream)
                img.load()  # force full decode now so a corrupt file fails here
            except Exception:
                error_message = "That file doesn't look like a valid image. Please upload a photo (PNG, JPG, etc.)."
            else:
                try:
                    prediction = run_inference(img)

                    # Save the upload so the preview box has something to show
                    # after the page reloads (the JS preview only lasts until submit).
                    saved_name = f"{uuid.uuid4().hex}.png"
                    img.convert("RGB").save(os.path.join(UPLOAD_DIR, saved_name))
                    img_path = f"/static/uploads/{saved_name}"
                except Exception as exc:  # noqa: BLE001 - surface the error to the user
                    error_message = f"Could not process that image: {exc}"

    return render_template(
        "index.html",
        prediction=prediction,
        img_path=img_path,
        error_message=error_message,
    )


@app.route("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
