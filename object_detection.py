from ultralytics import YOLO

print("Loading enhanced object detection model...")

model = YOLO("yolo11n.pt")

print("Object detection model loaded!")


def detect_objects(frame):

    # Higher image size helps with small and distant objects
    results = model(
        frame,
        conf=0.15,
        imgsz=1280,
        verbose=False
    )

    return results