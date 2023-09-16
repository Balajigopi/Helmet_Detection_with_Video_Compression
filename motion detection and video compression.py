import cv2

# Function to detect and capture motion in the video
def detect_and_capture_motion(video_path, output_path):
    cap = cv2.VideoCapture(video_path)

    # Check if the video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Define the background subtractor
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()

    # Video writer to save frames during motion
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise and improve background subtraction
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply background subtraction to detect motion
        fg_mask = bg_subtractor.apply(blurred)

        # Apply binary threshold to create a binary image
        _, thresholded = cv2.threshold(fg_mask, 20, 255, cv2.THRESH_BINARY)

        # Perform morphological operations to remove noise and fill gaps in the foreground
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        morphed = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)

        # Find contours of the moving objects
        contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Check if any significant motion is detected
        is_motion_detected = False
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Adjust this threshold based on your scenario
                is_motion_detected = True
                break

        if is_motion_detected:
            if out is None:
                out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))

            # Save frames during motion
            out.write(frame)
            cv2.putText(frame, "Motion Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            if out is not None:
                out.release()
                out = None

        # Display the motion detection result
        cv2.imshow('Motion Detection', frame)

        # Press 'q' to quit the program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if out is not None:
        out.release()

    cap.release()
    cv2.destroyAllWindows()

# Replace 'your_video_path.mp4' with the path to your video file
# Replace 'output_video.avi' with the desired output video filename
video_path = 'hellmet.mp4'
output_path = 'output_video.avi'
detect_and_capture_motion(video_path, output_path)
