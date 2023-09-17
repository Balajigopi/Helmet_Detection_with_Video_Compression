import cv2
import numpy as np

# Load the cascade for detecting helmets
helmet_cascade = cv2.CascadeClassifier('helmet_cascade.xml')

# Initialize video capture from video file
cap = cv2.VideoCapture('hellmet.mp4')

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Convert frame to grayscale for cascade detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect helmets in the frame
    helmets = helmet_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw bounding boxes around the detected helmets
    for (x, y, w, h) in helmets:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, 'Helmet', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame with helmet detections
    cv2.imshow('Helmet Tracker', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
