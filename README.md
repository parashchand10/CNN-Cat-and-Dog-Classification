# 🐱🐶 Cat & Dog Image Classification using CNN

## 🛠️ Tech Stack

### Programming
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

### Deep Learning
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)

### Data Processing
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)

### Web Application
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white)
![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white)

A Deep Learning web application that classifies uploaded images as **Cat** or **Dog** using a Convolutional Neural Network (CNN).

The trained CNN model is integrated with a **Flask web application**, allowing users to upload an image and receive a prediction with confidence scores.

---

## Project Overview

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

## Features

- Cat vs Dog image classification
- Custom CNN architecture
- Image upload through web interface
- Prediction confidence
- Real-time prediction using Flask
- Input image preprocessing
- Low-confidence image rejection
- Saved `.h5` trained model
- Simple browser-based interface

---

## CNN Model Architecture

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
