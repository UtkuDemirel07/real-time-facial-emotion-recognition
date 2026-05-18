import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def load_fer2013(csv_path, image_size=(48, 48)):
    data = pd.read_csv(csv_path)
    
    pixels = data['pixels'].tolist()
    width, height = image_size
    faces = []

    for pixel_sequence in pixels:
        face = np.array([int(p) for p in pixel_sequence.split()]).reshape(width, height)
        face = face.astype('float32') / 255.0  # Normalize
        faces.append(face)

    faces = np.asarray(faces)
    faces = np.expand_dims(faces, -1)  # (samples, 48, 48, 1)

    emotions = pd.get_dummies(data['emotion']).values

    return train_test_split(faces, emotions, test_size=0.2, random_state=42)

if __name__ == "__main__":
    csv_path = "data/fer2013.csv"
    X_train, X_test, y_train, y_test = load_fer2013(csv_path)
    print("Shapes:")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
