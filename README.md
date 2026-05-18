# Real-Time Facial Emotion Recognition

A real-time facial emotion recognition system built using Convolutional Neural Networks (CNNs), TensorFlow/Keras, and OpenCV.  
The project focuses on classifying human emotions from facial expressions using the FER-2013 dataset and deploying the trained model for live webcam-based emotion detection.

---

## Overview

This project was developed as part of a deep learning and computer vision study focused on facial emotion recognition (FER).  
The system uses grayscale facial images to classify seven different emotions and supports real-time webcam inference.

The project followed an iterative model development process:
- Baseline CNN implementation
- Improved CNN architecture with regularization
- Class balancing techniques
- Real-time deployment with OpenCV

---

## Features

- Real-time facial emotion detection
- Live webcam emotion prediction
- CNN-based deep learning pipeline
- Data augmentation techniques
- Batch normalization and dropout regularization
- Class weighting for imbalanced data
- Model checkpointing and early stopping
- Confusion matrix and performance evaluation
- Temporal smoothing for stable live predictions

---

## Emotion Classes

The model predicts the following emotions:

- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

---

## Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

---

## Dataset

This project uses the FER-2013 dataset containing 35,887 grayscale facial images sized 48x48 pixels.

Due to GitHub file size limitations, the dataset is not included in this repository.

You can download the dataset separately and place it inside the `data/` folder.

---

## Model Architecture

The final CNN architecture includes:

- Multiple stacked Conv2D layers
- ReLU activations
- Batch Normalization
- MaxPooling layers
- Dropout regularization
- Dense fully connected layers
- Softmax output layer

The final model was optimized using:
- Data augmentation
- Learning rate scheduling
- Class weighting
- Early stopping

---

## Results

| Metric | Final Model |
|---|---|
| Accuracy | 61.97% |
| Macro F1 Score | 0.5792 |
| Weighted F1 Score | 0.615 |

### Key Improvements
- Significant improvement in minority class recognition
- Better real-time stability
- Reduced overfitting
- Improved generalization

---

## Real-Time Inference

The trained model was integrated with OpenCV for real-time emotion recognition using webcam input.

The live system:
- Detects faces using Haar cascades
- Preprocesses facial regions
- Performs emotion prediction
- Displays live emotion labels
- Applies temporal smoothing for prediction stability

---

## Project Structure

```bash
facial_emotion_recognition/
│
├── data/
├── models/
├── notebooks/
├── screenshots/
├── src/
│   ├── train.py
│   ├── realtime_detection.py
│   ├── preprocessing.py
│   └── utils.py
│
├── requirements.txt
├── README.md
└── report.pdf
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/UtkuDemirel07/real-time-facial-emotion-recognition.git
cd real-time-facial-emotion-recognition
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run Real-Time Detection

```bash
python realtime_detection.py
```

---

## Future Improvements

- Lightweight mobile deployment
- Improved minority class balancing
- Better real-world robustness
- Multi-face tracking
- Transfer learning approaches
- Higher accuracy models

---

## Authors

- Ahmet Utku Demirel
- Liza Golmohammadi
- Pat Wantanasak
- Finn Walsh

---

## Academic Context

This project was developed as part of a university deep learning and computer vision research project focused on facial emotion recognition and real-time AI systems.

---

## License

This project is for educational and portfolio purposes.
