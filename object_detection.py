from ultralytics import YOLO

print("Loading balanced object detection model...")

model = YOLO("yolo11n.pt")

print("Object detection model loaded!")



IMPORTANT_OBJECTS = [
    "person",
    "backpack",
    "handbag",
    "suitcase",
    "book",
    "bottle",
    "cell phone",
    "laptop",
    "chair",
    "cup",
    "sports ball"
]


def detect_objects(frame):

    # Balanced settings:
    # Better distant detection but fewer false boxes
    results = model(
        frame,
        conf=0.35,
        iou=0.45,
        imgsz=1280,
        verbose=False
    )

    return results