import os
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load your model
model = load_model("cat_dog_cnn_model.h5")

REJECTION_THRESHOLD = 0.60      # used for single-label (binary / 3-class) models
MULTI_LABEL_THRESHOLD = 0.50    # used for 2-output multi-label models
IMG_SIZE = (150, 150)           # confirmed training input size


def predict_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    predictions = model.predict(img_array)[0]

    # Debug: check your terminal to see what your model actually outputs.
    print("Raw model output:", predictions, "shape:", predictions.shape)

    n_outputs = predictions.shape[0]

    if n_outputs == 1:
        # Binary sigmoid model: single value = P(dog). P(cat) = 1 - P(dog)
        dog_prob = float(predictions[0])
        cat_prob = 1.0 - dog_prob
        confidence = max(dog_prob, cat_prob)

        if confidence < REJECTION_THRESHOLD:
            return None

        winner = "dog" if dog_prob > cat_prob else "cat"
        return {"cat_pct": cat_prob * 100, "dog_pct": dog_prob * 100, "winner": winner}

    elif n_outputs == 2:
        # Multi-label model: independent probabilities [cat_prob, dog_prob].
        # This is the setup that actually supports detecting "both" in one image.
        cat_prob, dog_prob = float(predictions[0]), float(predictions[1])
        cat_present = cat_prob >= MULTI_LABEL_THRESHOLD
        dog_present = dog_prob >= MULTI_LABEL_THRESHOLD

        if not cat_present and not dog_present:
            return None

        winner = "both" if (cat_present and dog_present) else ("dog" if dog_present else "cat")
        return {"cat_pct": cat_prob * 100, "dog_pct": dog_prob * 100, "winner": winner}

    else:
        # 3-class softmax: [Cat, Dog, Both]
        cat_prob, dog_prob, both_prob = (float(p) for p in predictions[:3])
        confidence = max(cat_prob, dog_prob, both_prob)

        if confidence < REJECTION_THRESHOLD:
            return None

        if both_prob >= cat_prob and both_prob >= dog_prob:
            winner = "both"
        elif dog_prob >= cat_prob:
            winner = "dog"
        else:
            winner = "cat"

        return {"cat_pct": cat_prob * 100, "dog_pct": dog_prob * 100, "winner": winner}


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    img_path = None
    error_message = None

    if request.method == "POST":
        if "file" not in request.files:
            error_message = "No file uploaded."
            return render_template("index.html", error_message=error_message)

        file = request.files["file"]

        if file.filename == "":
            error_message = "No selected file."
            return render_template("index.html", error_message=error_message)

        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            result = predict_image(filepath)

            if result is None:
                error_message = "⚠️ Image not recognized! Please upload a valid picture of a cat, dog, or both."
            else:
                prediction = result
                img_path = filepath

    return render_template("index.html", prediction=prediction, img_path=img_path, error_message=error_message)


if __name__ == "__main__":
    app.run(debug=True, port=5001)