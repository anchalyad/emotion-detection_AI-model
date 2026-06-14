import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import load_model

data = []
labels = []

emotions = ['Angry','Happy','Neutral','Sad','Surprise']


for i, emotion in enumerate(emotions):
    path = "data/" + emotion
    
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        face = cv2.resize(gray, (48,48))
        face = face / 255.0
        
        data.append(face)
        labels.append(i)

# Convert to numpy
data = np.array(data)
labels = np.array(labels)

# Reshape
data = data.reshape(-1,48,48,1)

labels = to_categorical(labels, num_classes=7)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)



model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dense(7, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

model.save("my_model.keras")

img_path = "Data/Happy/happy.jpg"

img = cv2.imread(img_path)

if img is None:
    print("Error: Image not found")
else:
    print("Image loaded successfully")


model.save("my_model.keras")

# Preprocess
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
face = cv2.resize(gray, (48,48))
face = face / 255.0
face = face.reshape(1,48,48,1)

# Predict
prediction = model.predict(face)
print(prediction)

# Get result
emotion = emotions[np.argmax(prediction)]

print("Predicted Emotion:", emotion)




