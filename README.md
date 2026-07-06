# Face Mask Detector

A real-time face mask detector: a Keras CNN trained to classify "mask" vs. "no mask," wired up to OpenCV for live webcam inference with on-screen bounding boxes.

## What this covers
This project practices the combination of a trained image classifier with classical computer vision for real-time inference — using a Haar Cascade to locate faces in a video frame, cropping each detected face, and running it through a CNN classifier every frame. It's a solid talking point for discussing the difference between face *detection* (Haar Cascade) and face mask *classification* (the trained CNN), and how the two are chained together in a live pipeline.

## Tech Stack
- Python, Keras/TensorFlow (CNN training and inference)
- OpenCV (`cv2`) — Haar Cascade face detection, webcam capture, live annotation
- NumPy

## Approach
- **Model architecture**: 3 Conv2D+MaxPooling blocks (32 filters each) on 150x150x3 input, followed by a Dense(100) layer and a single sigmoid output for binary mask/no-mask classification.
- **Training**: `ImageDataGenerator` with rescaling, shear, zoom, and horizontal flip augmentation on the training set; trained with the Adam optimizer and binary cross-entropy loss for 10 epochs, batch size 16. Trained weights are saved to `mymodel.h5` (included in this repo).
- **Face detection**: `haarcascade_frontalface_default.xml` (OpenCV's pretrained Haar Cascade) locates face bounding boxes in each webcam frame before classification.
- **Live inference loop**: for each detected face, the crop is resized to 150x150, passed through the CNN, and the frame is annotated with a green ("MASK") or red ("NO MASK") bounding box plus a timestamp overlay, using `cv2.imshow` for a live video window.
- **Also supports single-image prediction** — loading one image file and getting a mask/no-mask classification without the webcam loop.
- `test.zip` in this repo holds a small set of sample test images for the with-mask/without-mask classes.

## Running Locally
```bash
git clone https://github.com/vinay23is/Face_Mask_Detector.git
cd Face_Mask_Detector
pip install tensorflow keras opencv-python numpy
python facemask.py
```
By default the script loads the pretrained `mymodel.h5` and opens your default webcam (`cv2.VideoCapture(0)`) for live detection — press `q` to quit. The training code (CNN definition + `ImageDataGenerator`) is included at the top of `facemask.py`; uncomment it and point it at your own `train`/`test` folders (structured as `<mask>/<no_mask>` subfolders) to retrain from scratch instead of using the bundled model.
