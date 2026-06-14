import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Load CSV
df = pd.read_csv("dataset/fer2013.csv")

# Convert pixel string → array
def process_pixels(pixels):
    pixels = np.array(pixels.split(), dtype="float32")
    return pixels.reshape(48, 48, 1)

# X and y
X = np.array(df["pixels"].apply(process_pixels).tolist()) / 255.0
y = to_categorical(df["emotion"], num_classes=7)

# Model
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(48,48,1)),
    MaxPooling2D((2,2)),

    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D((2,2)),

    Flatten(),

    Dense(128, activation="relu"),
    Dense(7, activation="softmax")
])

# Compile
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Train
model.fit(X, y, epochs=20, batch_size=32)

# Save
model.save("my_emotion_model.h5")