# Cat vs. Dog Classifier — Flask App

A small Flask web app that serves the `cat_dog_cnn_model_v2.keras` model
(trained in `Fine_Tune.ipynb`) for real-time cat/dog image predictions.

## Project structure

```
cat-dog-classifier/
├── app.py                # Flask app + Keras (.keras) inference
├── requirements.txt
├── Procfile               # for Render / Heroku (gunicorn)
├── templates/
│   └── index.html         # PetVision AI frontend
├── static/
│   ├── style.css          # unused now (index.html has its own <style>) — safe to delete
│   ├── dog.png             # <-- you need to add this: page background image
│   └── uploads/            # uploaded images are saved here for preview
└── model/
    └── cat_dog_cnn_model_v2.keras   # <-- you need to add this file
```

## 1. Add the model file and background image

This repo does **not** include the trained model. Grab the file already
produced in your notebook (`Fine_Tune.ipynb`, section "11. Save the Final
Model" — `model.save("cat_dog_cnn_model_v2.keras")`) and drop it here:

```
model/cat_dog_cnn_model_v2.keras
```

No conversion step is needed for this version — it loads the `.keras`
file directly with `tf.keras.models.load_model()`.

The template also expects a background image at `static/dog.png` — add any
image with that filename, or edit the `body { background: url(...) }` rule
in `templates/index.html` to point elsewhere.

**Note on deploy size:** this version depends on full `tensorflow`, which
is much heavier than the TFLite runtime (hundreds of MB vs a few MB). On
Render's free tier this means slower builds and higher memory use at
runtime — worth knowing if you hit build timeouts or out-of-memory errors.

## 2. Run locally

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Visit `http://localhost:5000` and upload an image.

## 3. Deploy on Render

1. Push this folder to a GitHub repo.
2. On Render: **New → Web Service** → connect the repo.
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn app:app`
5. Make sure `model/cat_dog_cnn_model_v2.keras` is actually committed to
   the repo as a real binary (not a Git LFS pointer — check its file size
   after cloning) — Render's filesystem is ephemeral otherwise and won't
   have your model on deploy.

## Notes

- Input images are resized to 150×150 to match training (`IMG_SIZE` in the
  notebook).
- Preprocessing intentionally does **not** rescale by /255 — the
  MobileNetV2 `preprocess_input` normalization is baked into the model
  graph itself, so raw resized pixel values are passed straight to
  `model.predict()` (matches the notebook's own inference cell).
