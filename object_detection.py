from ultralytics import YOLO

print("Loading object detection model...")

object_model = YOLO("yolo11n.pt")


def detect_objects(frame):
    results = object_model(
        frame,
        conf=0.30,
        verbose=False
    )

    return results