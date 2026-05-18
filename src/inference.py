from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os

model_path = 'checkpoints/best_model.h5'
model = load_model(model_path)

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def preprocess_image(image_path):
    """Görseli model için hazırla."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  
    img = cv2.resize(img, (48, 48))                     
    img = img.astype('float32') / 255.0                 
    img = np.expand_dims(img, axis=-1)                  
    img = np.expand_dims(img, axis=0)                   
    return img

def predict_emotion(image_path):
    """Bir resimde duyguyu tahmin et."""
    img = preprocess_image(image_path)
    predictions = model.predict(img)
    emotion_idx = np.argmax(predictions)
    emotion = emotion_labels[emotion_idx]
    return emotion

if __name__ == "__main__":
    # Test için örnek bir fotoğraf yolu
    test_image_path = 'test/fear/PrivateTest_902573.jpg'
  
    if os.path.exists(test_image_path):
        predicted_emotion = predict_emotion(test_image_path)
        print(f"Tahmin edilen duygu: {predicted_emotion}")
    else:
        print("Test görseli bulunamadı!")
