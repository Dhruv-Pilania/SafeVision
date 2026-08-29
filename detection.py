from ultralytics import YOLO


print("Loading person detection model...")

model = YOLO("yolo11n.pt")

print("Person detection model loaded!")


def detect_people(frame):

    results = model(
        frame,
        classes=[0],
        conf=0.35,
        verbose=False
    )

    return results