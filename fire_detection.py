from ultralytics import YOLO

print("Loading fire and smoke model...")

fire_model = YOLO("fire-smoke.pt")

def detect_fire_smoke(frame):
    results = fire_model(
        frame,
        conf=0.30,
        verbose=False
    )
    return results