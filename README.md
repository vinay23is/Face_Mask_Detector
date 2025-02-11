# Face Mask Detection

This repository contains a Python script for detecting face masks using deep learning.

## Files in this Repository

- **`facemask.py`**: The main script that detects whether a person is wearing a mask or not.
- **`haarcascade_frontalface_default.xml`**: A pre-trained Haar Cascade classifier used for face detection.
- **`mymodel.h5`**: A trained deep learning model for classifying masked and unmasked faces.
- **`README.md`**: This documentation file explaining the project.

## Features

- **Train a CNN Model**: The script includes a CNN model built with Keras to classify images of faces as wearing a mask or not.
- **Pretrained Model**: It loads a pre-trained model (`mymodel.h5`) to detect face masks in real-time.
- **Live Video Mask Detection**: Uses OpenCV to capture video, detect faces, and predict whether a mask is worn.
- **Image-based Prediction**: Can classify whether an individual image contains a masked or unmasked face.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/face-mask-detection.git
   cd face-mask-detection
