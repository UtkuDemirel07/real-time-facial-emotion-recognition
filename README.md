# Real-Time Facial Emotion Recognition

A deep learning based facial emotion recognition system built using Convolutional Neural Networks (CNNs), OpenCV, and the FER-2013 dataset.

The project includes:
- Custom CNN architecture
- Real-time webcam emotion detection
- Data augmentation
- Batch normalization and dropout
- Class balancing strategies
- Model comparison and optimisation
- Live facial emotion inference

## Technologies Used
- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
- CNNs

## Dataset
FER-2013 dataset containing 35,887 grayscale facial images across 7 emotion categories:
- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

## Final Results
- Final Accuracy: 61.97%
- Macro F1 Score: 0.5792
- Real-time webcam inference implemented
- Improved minority emotion recognition using class weighting and augmentation

## Features
- Real-time face detection
- Emotion prediction smoothing
- CNN model optimisation
- Confusion matrix analysis
- Data augmentation pipeline
- Model checkpointing and early stopping
