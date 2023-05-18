import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + './haarcascade_frontalface_default.xml')

def detect_faces(image):
    # Convert the image to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Save the recognized faces
    saved_faces = []
    for i, (x, y, w, h) in enumerate(faces):
        face_image = image[y:y+h, x:x+w]  # Extract the face region
        face_path = f'./images/face{i}.jpg'  # Path to save the face image
        cv2.imwrite(face_path, face_image)  # Save the face image
        saved_faces.append(face_path)  # Add the face image path to the list

    return saved_faces

def run(image_path):
    image = cv2.imread(image_path)
    saved_faces = detect_faces(image)
    print("Recognized faces saved to the /images directory:")
    for face_path in saved_faces:
        print(face_path)

run('./images/linkin_park.jpg')

