from data_loader import load_fer2013
from model import build_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import os

# Veriyi yükle
csv_path = 'data/fer2013.csv'
X_train, X_test, y_train, y_test = load_fer2013(csv_path)

# Modeli oluştur
input_shape = (48, 48, 1)
num_classes = 7
model = build_model(input_shape, num_classes)

# Modeli derle
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Modeli kaydetmek için checkpoint
checkpoint_dir = 'checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

checkpoint = ModelCheckpoint(
    filepath=os.path.join(checkpoint_dir, 'best_model.h5'),
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# Eğit
model.fit(
    X_train, y_train,
    batch_size=64,
    epochs=25,
    validation_data=(X_test, y_test),
    callbacks=[checkpoint]
)
