# 🐱🐶 Cat & Dog Image Classification using CNN

A Deep Learning web application that classifies uploaded images as **Cat** or **Dog** using a Convolutional Neural Network (CNN).

The trained CNN model is integrated with a **Flask web application**, allowing users to upload an image and receive a prediction with confidence scores.

---

## 🚀 Project Overview

This project demonstrates an end-to-end Deep Learning workflow:

- Dataset preparation
- Image preprocessing
- CNN model development
- Model training and validation
- Model evaluation
- Model saving
- Flask web application integration
- Image upload and prediction
- Prediction confidence display

The model accepts images resized to **150 × 150 pixels** and normalizes pixel values to the range **0–1** before prediction.

---

## ✨ Features

- 🐱 Cat vs Dog image classification
- 🧠 Custom CNN architecture
- 📤 Image upload through web interface
- 📊 Prediction confidence
- ⚡ Real-time prediction using Flask
- 🔍 Input image preprocessing
- ⚠️ Low-confidence image rejection
- 💾 Saved `.h5` trained model
- 🌐 Simple browser-based interface

---

## 🧠 CNN Model Architecture

The model uses a Sequential CNN architecture:

```text
Input Image
   │
   ▼
150 × 150 × 3
   │
   ▼
Conv2D - 32 Filters
   │
   ▼
MaxPooling2D
   │
   ▼
Conv2D - 64 Filters
   │
   ▼
MaxPooling2D
   │
   ▼
Conv2D - 128 Filters
   │
   ▼
MaxPooling2D
   │
   ▼
Flatten
   │
   ▼
Dense - 512 Neurons
   │
   ▼
Dropout - 0.5
   │
   ▼
Dense - 1 Neuron
   │
   ▼
Sigmoid
   │
   ▼
Cat / Dog
