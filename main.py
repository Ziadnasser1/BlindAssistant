import cv2
import pyttsx3
import threading
import time
from queue import Queue
from ultralytics import YOLO

KNOWN_WIDTHS = {
    'person': 40.0,
    'bicycle': 60.0,
    'car': 180.0,
    'motorcycle': 80.0,
    'airplane': 2000.0,
    'bus': 250.0,
    'train': 300.0,
    'truck': 250.0,
    'boat': 250.0,
    'traffic light': 30.0,
    'fire hydrant': 20.0,
    'stop sign': 75.0,
    'parking meter': 20.0,
    'bench': 120.0,
    'bird': 20.0,
    'cat': 30.0,
    'dog': 50.0,
    'horse': 150.0,
    'sheep': 100.0,
    'cow': 150.0,
    'elephant': 300.0,
    'bear': 150.0,
    'zebra': 150.0,
    'giraffe': 200.0,
    'backpack': 30.0,
    'umbrella': 100.0,
    'handbag': 30.0,
    'tie': 10.0,
    'suitcase': 50.0,
    'frisbee': 30.0,
    'skis': 10.0,
    'snowboard': 30.0,
    'sports ball': 22.0,
    'kite': 100.0,
    'baseball bat': 7.0,
    'baseball glove': 20.0,
    'skateboard': 20.0,
    'surfboard': 50.0,
    'tennis racket': 30.0,
    'bottle': 7.0,
    'wine glass': 5.0,
    'cup': 8.0,
    'fork': 2.5,
    'knife': 2.0,
    'spoon': 3.0,
    'bowl': 15.0,
    'banana': 3.0,
    'apple': 8.0,
    'sandwich': 15.0,
    'orange': 8.0,
    'broccoli': 15.0,
    'carrot': 2.5,
    'hot dog': 5.0,
    'pizza': 30.0,
    'donut': 10.0,
    'cake': 20.0,
    'chair': 50.0,
    'couch': 150.0,
    'potted plant': 30.0,
    'bed': 160.0,
    'dining table': 100.0,
    'toilet': 40.0,
    'tv': 100.0,
    'laptop': 35.0,
    'mouse': 6.0,
    'remote': 5.0,
    'keyboard': 45.0,
    'cell phone': 7.0,
    'microwave': 50.0,
    'oven': 60.0,
    'toaster': 30.0,
    'sink': 50.0,
    'refrigerator': 70.0,
    'book': 15.0,
    'clock': 30.0,
    'vase': 15.0,
    'scissors': 6.0,
    'teddy bear': 20.0,
    'hair drier': 8.0,
    'toothbrush': 2.0
}

FOCAL_LENGTH = 700

engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 0.9)


def speak(text_queue):
    while True:
        text = text_queue.get()
        if text is None:
            break
        engine.say(text)
        engine.runAndWait()
        text_queue.task_done()


model = YOLO('yolov8n.pt')

coco_classes = model.names

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video device.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_skip = 4
frame_count = 0

text_queue = Queue()

tts_thread = threading.Thread(target=speak, args=(text_queue,))
tts_thread.start()

last_announcement_time = time.time()
announcement_interval = 5

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        start_time = time.time()

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame_rgb, stream=True)

        current_time = time.time()
        objects_detected = False
        # distance=1

        if current_time - last_announcement_time >= announcement_interval:
            for result in results:
                for pred in result.boxes:
                    x1, y1, x2, y2 = map(int, pred.xyxy[0])
                    conf = pred.conf[0]
                    cls = int(pred.cls[0])
                    label = coco_classes[cls]

                    if label in KNOWN_WIDTHS:
                        width_in_image = x2 - x1
                        known_width = KNOWN_WIDTHS[label]

                        distance = (known_width * FOCAL_LENGTH) / width_in_image

                        speech_text = f'There is a {label} at a distance of {distance:.2f} centimeters'
                        text_queue.put(speech_text)
                        last_announcement_time = current_time
                        objects_detected = True
                        break
        else:
            for result in results:
                for pred in result.boxes:
                    x1, y1, x2, y2 = map(int, pred.xyxy[0])
                    conf = pred.conf[0]
                    cls = int(pred.cls[0])
                    label = coco_classes[cls]

                    if label in KNOWN_WIDTHS:
                        width_in_image = x2 - x1
                        known_width = KNOWN_WIDTHS[label]

                        distance = (known_width * FOCAL_LENGTH) / width_in_image

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f'{label} {distance:.2f} cm', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                    (36, 255, 12), 2)

        end_time = time.time()
        fps = 1 / ((end_time - start_time) + 1)
        cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow('YOLO Object Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    text_queue.put(None)
    tts_thread.join()
