from ultralytics import YOLO
import time


print("Loading pose model...")

pose_model = YOLO("yolo11n-pose.pt")

print("Pose model loaded!")


fall_start_time = None
FALL_CONFIRMATION_TIME = 30


def detect_fall(frame):

    global fall_start_time

    results = pose_model(
        frame,
        conf=0.35,
        verbose=False
    )

    fallen_person = False

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            width = x2 - x1
            height = y2 - y1

            # A lying person's bounding box often becomes wider
            # than it is tall
            if width > height * 1.2:

                fallen_person = True

    if fallen_person:

        if fall_start_time is None:
            fall_start_time = time.time()

        elapsed = time.time() - fall_start_time

        if elapsed >= FALL_CONFIRMATION_TIME:
            return True, elapsed

        return False, elapsed

    else:

        fall_start_time = None
        return False, 0