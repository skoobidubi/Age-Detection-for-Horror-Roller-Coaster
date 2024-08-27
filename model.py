#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# train_model.ipynb

# Importing required libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# Load the dataset (replace with your dataset)
# Dataset should be preprocessed to include images, age, and gender labels
data = pd.read_csv('path_to_dataset.csv')

# Preprocess the dataset (assume images are already resized and labels are encoded)
X = np.array(data['images'].tolist())
y_age = np.array(data['age'].tolist())
y_gender = np.array(data['gender'].tolist())

# Split the data into training and testing sets
X_train, X_test, y_age_train, y_age_test = train_test_split(X, y_age, test_size=0.2, random_state=42)
X_train, X_test, y_gender_train, y_gender_test = train_test_split(X, y_gender, test_size=0.2, random_state=42)

# Build the Age Detection Model
age_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='linear')  # Regression output for age
])

# Compile the model
age_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the model
history_age = age_model.fit(X_train, y_age_train, epochs=10, validation_split=0.1)

# Evaluate the model
age_preds = age_model.predict(X_test)
print(f'Age Model Mean Absolute Error: {np.mean(np.abs(y_age_test - age_preds))}')

# Build the Gender Detection Model
gender_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification output for gender
])

# Compile the model
gender_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history_gender = gender_model.fit(X_train, y_gender_train, epochs=10, validation_split=0.1)

# Evaluate the model
gender_preds = (gender_model.predict(X_test) > 0.5).astype("int32")
print(f'Gender Model Accuracy: {accuracy_score(y_gender_test, gender_preds)}')

# Save models
age_model.save('age_model.h5')
gender_model.save('gender_model.h5')

# Plot training history
plt.plot(history_age.history['mae'], label='Age MAE')
plt.plot(history_gender.history['accuracy'], label='Gender Accuracy')
plt.legend()
plt.show()

# Confusion Matrix for Gender Model
conf_matrix = confusion_matrix(y_gender_test, gender_preds)
print(conf_matrix)


# In[ ]:


import tkinter as tk
from tkinter import Label
import cv2
from PIL import Image, ImageTk

# Function to update the frame
def update_frame():
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        lbl.imgtk = imgtk
        lbl.configure(image=imgtk)
    lbl.after(10, update_frame)

# Initialize GUI window
root = tk.Tk()
root.title("Horror Roller Coaster Age Detection")
lbl = Label(root)
lbl.pack()

# Initialize video capture
cap = cv2.VideoCapture(0)

# Start the video stream
update_frame()
root.mainloop()

# Release the capture when the window is closed
cap.release()
cv2.destroyAllWindows()

