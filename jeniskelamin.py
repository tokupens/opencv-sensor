import cv2

# Load Haar Cascade untuk deteksi wajah
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load model klasifikasi jenis kelamin
gender_net = cv2.dnn.readNet('synthetic_leftright_balanced_model.xml')

def classify_gender(face):
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = "Male" if gender_preds[0, 0] > gender_preds[0, 1] else "Female"
    return gender

# Open video capture (0 for default camera)
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face region
        face = frame[y:y+h, x:x+w]

        # Classify gender
        gender = classify_gender(face)

        # Draw bounding box and gender on the frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, f'Gender: {gender}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Gender Sensor', frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
