import os
from flask import Flask, render_template, request
from PIL import Image
import numpy as np
from ai_edge_litert.interpreter import Interpreter

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load TFLite Model
MODEL_PATH = "cat_dog_cnn_model.tflite"
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

REJECTION_THRESHOLD = 0.60
MULTI_LABEL_THRESHOLD = 0.50
IMG_SIZE = (150, 150)


def predict_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]

    n_outputs = predictions.shape[0]

    if n_outputs == 1:
        dog_prob = float(predictions[0])
        cat_prob = 1.0 - dog_prob
        confidence = max(dog_prob, cat_prob)

        if confidence < REJECTION_THRESHOLD:
            return None

        winner = "dog" if dog_prob > cat_prob else "cat"
        return {"cat_pct": cat_prob * 100, "dog_pct": dog_prob * 100, "winner": winner}

    elif n_outputs == 2:
        cat_prob, dog_prob = float(predictions[0]), float(predictions[1])
        cat_present = cat_prob >= MULTI_LABEL_THRESHOLD
        dog_present = dog_prob >= MULTI_LABEL_THRESHOLD

        if not cat_present and not dog_present:
            return None

        winner = "both" if (cat_present and dog_present) else ("dog" if dog_present else "cat")
        return {"cat_pct": cat_prob * 100, "dog_pct": dog_prob * 100, "winner": winner}

    else:
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
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)
