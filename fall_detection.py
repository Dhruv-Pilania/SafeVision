from ultralytics import YOLO
import time
import math


print("Loading enhanced pose model...")

pose_model = YOLO("yolo11n-pose.pt")

print("Enhanced pose model loaded!")


# =====================================================
# MEMORY
# =====================================================

fall_start_time = None
previous_center_y = None
previous_time = None


# =====================================================
# SETTINGS
# =====================================================

FALL_CONFIRMATION_TIME = 3.0

# Person must become significantly horizontal
LYING_RATIO = 1.25

# Required confidence
MIN_CONFIDENCE = 0.45


def distance(point1, point2):

    return math.sqrt(
        (point2[0] - point1[0]) ** 2 +
        (point2[1] - point1[1]) ** 2
    )


def detect_fall(frame):

    global fall_start_time
    global previous_center_y
    global previous_time

    results = pose_model(
        frame,
        conf=MIN_CONFIDENCE,
        verbose=False
    )

    current_time = time.time()

    fallen_person_detected = False


    for result in results:

        if result.boxes is None:
            continue

        if result.keypoints is None:
            continue


        for index, box in enumerate(result.boxes):

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            width = x2 - x1
            height = y2 - y1

            box_confidence = float(box.conf[0])


            # =========================================
            # CONDITION 1:
            # PERSON MUST LOOK HORIZONTAL
            # =========================================

            is_horizontal = (
                width > height * LYING_RATIO
            )


            # =========================================
            # GET BODY CENTER
            # =========================================

            center_y = (
                y1 + y2
            ) / 2


            # =========================================
            # CONDITION 2:
            # CHECK BODY POSITION CHANGE
            # =========================================

            sudden_downward_movement = False

            if (
                previous_center_y is not None
                and previous_time is not None
            ):

                time_difference = (
                    current_time - previous_time
                )

                if time_difference > 0:

                    movement_speed = (
                        center_y -
                        previous_center_y
                    ) / time_difference

                    # Large downward movement
                    if movement_speed > 250:

                        sudden_downward_movement = True


            # Save current position
            previous_center_y = center_y
            previous_time = current_time


            # =========================================
            # FALL DETECTION LOGIC
            # =========================================

            # A person is considered fallen only if:
            #
            # 1. The body is horizontal
            # AND
            # 2. A sudden downward movement happened
            #
            # OR the person remains strongly horizontal

            if (
                is_horizontal
                and box_confidence >= MIN_CONFIDENCE
            ):

                fallen_person_detected = True


    # =================================================
    # FALL TIMER
    # =================================================

    if fallen_person_detected:

        if fall_start_time is None:

            fall_start_time = current_time

        elapsed_time = (
            current_time -
            fall_start_time
        )


        # Final alert after 3 seconds
        if elapsed_time >= FALL_CONFIRMATION_TIME:

            return True, elapsed_time

        return False, elapsed_time


    # =================================================
    # RESET
    # =================================================

    else:

        fall_start_time = None

        return False, 0