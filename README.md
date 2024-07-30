## Blind Assistant Project

### Overview
This project aims to develop a real-time object detection and distance estimation application to assist visually impaired individuals. The system uses a laptop camera to capture the surroundings and provides auditory feedback about detected objects and their distances.

### Features
- **Real-Time Object Detection**: Utilizes the YOLO (You Only Look Once) model for fast and accurate object detection.
- **Auditory Feedback**: Announces the identities and distances of detected objects using text-to-speech (TTS) through the pyttsx3 library.
- **Distance Estimation**: Estimates the distance of objects from the camera based on their known physical widths and the focal length of the camera.
- **User-Friendly Interface**: Displays the detected objects and distances on the screen with bounding boxes and labels, providing visual feedback for those with partial vision.

### Technical Details
- **YOLO Model**: Uses the YOLOv8 model from the Ultralytics library for object detection.
- **COCO Dataset**: Supports object detection for the 80 classes defined in the COCO dataset.
- **OpenCV**: Handles video capture and processing, converting frames to the required format for the YOLO model.
- **Threading**: Uses a separate thread for text-to-speech to ensure smooth and uninterrupted operation.
- **Configuration**: Adjustable properties like frame skip and announcement intervals to balance performance and responsiveness.

### How It Works
1. **Capture Frame**: The application captures frames from the laptop's webcam.
2. **Object Detection**: Each frame is processed by the YOLO model to detect objects.
3. **Distance Calculation**: For each detected object, the application calculates its distance from the camera using the known physical width of the object and the camera's focal length.
4. **Auditory Feedback**: The application announces the detected objects and their distances through TTS.
5. **Visual Feedback**: The detected objects are highlighted on the video feed with bounding boxes and distance labels.

### Usage
1. **Setup**: Install the required libraries (`cv2`, `pyttsx3`, `threading`, `queue`, `ultralytics`).
2. **Run the Application**: Execute the Python script to start the real-time object detection and auditory feedback system.
3. **Interact**: The system will continuously provide updates on detected objects and their distances. Press 'q' to exit the application.

### Conclusion
This project enhances the independence and mobility of visually impaired individuals by providing real-time information about their surroundings. By leveraging state-of-the-art object detection and text-to-speech technologies, it offers a practical and accessible solution for everyday use.
