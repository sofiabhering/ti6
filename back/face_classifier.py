import cv2
import tensorflow as tf
from tensorflow.keras import models
import PIL
import os
import numpy as np

img_coords = {}
img_classes = {}


def detect_faces(image):
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + './haarcascade_frontalface_default.xml')
    # Convert the image to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5)

    # Save the recognized faces
    saved_faces = []
    for i, (x, y, w, h) in enumerate(faces):
        img_coords[i] = (x, y, w, h)
        face_image = image[y:y+h, x:x+w]  # Extract the face region
        face_image = cv2.resize(face_image, (200, 200))
        face_path = f'./images/face{i}.jpg'  # Path to save the face image
        cv2.imwrite(face_path, face_image)  # Save the face image


def blur(image):
    processed_img = image.copy()
    for i in img_coords:
        if img_classes[i] == 0:
            x, y, w, h = img_coords[i]
            region = processed_img[y:y+h, x:x+w]
            black_square = np.zeros((h, w, 3), dtype=np.uint8)
            # blurred_region = cv2.GaussianBlur(region, (25, 25), 0)
            processed_img[y:y+h, x:x+w] = black_square 

    cv2.imwrite('./uploads/processed_image.jpg', processed_img)


def predict():
    model = models.load_model('./datasets/models/test-10-0.797-0.814.model/')
    for i, file in enumerate(os.listdir('./images')):
        img_array = PIL.Image.open('./images/'+file)
        img_array = tf.expand_dims(img_array, 0)
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        img_classes[i] = np.argmax(score)
        print(f"{file} most likely belongs to {np.argmax(score)} with a {100 * np.max(score)} percent confidence.")


def run(image_path):
    image = cv2.imread(image_path)
    detect_faces(image)
    predict()
    blur(image)


run('./uploads/linkin_park.jpg')
