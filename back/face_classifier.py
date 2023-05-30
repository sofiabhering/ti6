import cv2
import tensorflow as tf
import PIL
import os
import numpy as np
from multiprocessing import Pool, current_process
from functools import partial

import sys

img_coords = {}
img_classes = {}

model = tf.keras.models.load_model(
    './datasets/models/test-05-0.941-0.919.model/', compile=False)
model.compile(optimizer='adam',
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])


def detect_faces(image):
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + './haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5)

    for i, (x, y, w, h) in enumerate(faces):
        img_coords[f'face{i}'] = (x, y, w, h)
        face_image = image[y:y+h, x:x+w]  # Extract the face region
        face_image = cv2.resize(face_image, (200, 200))
        face_path = f'./images/face{i}.jpg'  # Path to save the face image
        cv2.imwrite(face_path, face_image)  # Save the face image


def blur(image, name):
    processed_img = image.copy()
    for i in img_coords:
        if img_classes[i] == 0:
            x, y, w, h = img_coords[i]
            region = processed_img[y:y+h, x:x+w]
            black_square = np.zeros((h, w, 3), dtype=np.uint8)
            # blurred_region = cv2.GaussianBlur(region, (25, 25), 0)
            processed_img[y:y+h, x:x+w] = black_square

    cv2.imwrite(f'../front-end/front-end/uploads/processed_{name}', processed_img)


def process_image(file, model, process_name):
    img_array = PIL.Image.open('./images/' + file)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array, use_multiprocessing=True, verbose=1)
    score = tf.nn.softmax(predictions[0])
    return file, np.argmax(score), np.max(score), process_name


def predict(model, files):
    print('\t-processing-')

    with Pool() as pool:
        func = partial(process_image, model=model, process_name=current_process().name)
        results = pool.map(func, files)

    for file, class_index, confidence, process_name in results:
        print(f"{file} (processed by {process_name}) most likely belongs to {class_index} with a {100 * confidence} percent confidence.")
        img_classes[file[:-4]] = class_index


def run(image_path, name):
    print(f'image_path -> {image_path}', file=sys.stdout)
    image = cv2.imread(image_path)
    detect_faces(image)

    files = os.listdir('./images')
    predict(model, files)
    blur(image, name)

    for file in os.listdir('./images'):
        os.remove(f'./images/{file}')


# name = 'test2.jpg'
# if __name__ == '__main__':
#     run(f'./uploads/{name}',name)
