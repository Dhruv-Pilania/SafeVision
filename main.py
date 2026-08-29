import cv2
import time

from device import open_camera, read_frame, close_camera
from object_detection import detect_objects
from crowd_detection import detect_crowd_from_people
from fall_detection import detect_fall
from fire_detection import detect_fire_smoke


# =====================================================
# OBJECT COLOURS
# =====================================================

PERSON_COLOR = (0, 255, 0)
OBJECT_COLOR = (255, 0, 0)
FIRE_COLOR = (0, 0, 255)
SMOKE_COLOR = (128, 128, 128)


# =====================================================
# MAIN SYSTEM
# =====================================================

def main():

    print("=" * 50)
    print("SENTINE AI SAFETY SYSTEM")
    print("=" * 50)

    cap = open_camera(0)

    if cap is None:
        print("ERROR: Could not open camera.")
        return

    print("Camera started successfully!")
    print("Press Q to close.")


    while True:

        # =============================================
        # READ CAMERA
        # =============================================

        frame = read_frame(cap)

        if frame is None:
            print("ERROR: Could not read frame.")
            break


        # =============================================
        # RESIZE FOR BETTER PROCESSING
        # =============================================

        original_height, original_width = frame.shape[:2]

        frame = cv2.resize(
            frame,
            (1280, 720)
        )


        # =============================================
        # OBJECT DETECTION
        # =============================================

        person_count = 0
        detected_objects = []

        try:

            object_results = detect_objects(frame)

            for result in object_results:

                if result.boxes is None:
                    continue

                for box in result.boxes:

                    # Get coordinates
                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0].tolist()
                    )

                    # Get confidence
                    confidence = float(box.conf[0])

                    # Get class
                    class_id = int(box.cls[0])

                    # Get object name
                    class_name = model_name(
                        result,
                        class_id
                    )

                    class_name = class_name.lower()


                    # ---------------------------------
                    # COUNT PEOPLE
                    # ---------------------------------

                    if class_name == "person":

                        person_count += 1

                        color = PERSON_COLOR

                    else:

                        color = OBJECT_COLOR


                    # ---------------------------------
                    # SAVE DETECTED OBJECT
                    # ---------------------------------

                    detected_objects.append(
                        class_name
                    )


                    # ---------------------------------
                    # DRAW BOX
                    # ---------------------------------

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        color,
                        2
                    )


                    # ---------------------------------
                    # OBJECT LABEL
                    # ---------------------------------

                    label = (
                        f"{class_name.upper()} "
                        f"{confidence:.0%}"
                    )

                    cv2.putText(
                        frame,
                        label,
                        (x1, max(30, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2
                    )


        except Exception as e:

            print("Object detection error:", e)


        # =============================================
        # CROWD DETECTION
        # =============================================

        crowd_detected = detect_crowd_from_people(
            person_count,
            crowd_threshold=5
        )


        # =============================================
        # FALL DETECTION
        # =============================================

        fall_detected = False
        fall_time = 0

        try:

            fall_result = detect_fall(frame)

            # Support your previous fall module format
            if isinstance(fall_result, tuple):

                fall_detected = fall_result[0]

                if len(fall_result) > 1:
                    fall_time = fall_result[1]

            elif isinstance(fall_result, dict):

                fall_detected = fall_result.get(
                    "fall_detected",
                    False
                )

            else:

                fall_detected = bool(fall_result)


        except Exception as e:

            print("Fall detection error:", e)


        # =============================================
        # FIRE AND SMOKE DETECTION
        # =============================================

        fire_detected = False
        smoke_detected = False

        try:

            fire_results = detect_fire_smoke(frame)

            for result in fire_results:

                if result.boxes is None:
                    continue

                for box in result.boxes:

                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0].tolist()
                    )

                    confidence = float(box.conf[0])

                    class_id = int(box.cls[0])

                    class_name = model_name(
                        result,
                        class_id
                    ).lower()


                    # ---------------------------------
                    # FIRE
                    # ---------------------------------

                    if "fire" in class_name:

                        fire_detected = True

                        cv2.rectangle(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            FIRE_COLOR,
                            4
                        )

                        cv2.putText(
                            frame,
                            f"FIRE {confidence:.0%}",
                            (x1, max(40, y1 - 15)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            FIRE_COLOR,
                            3
                        )


                    # ---------------------------------
                    # SMOKE
                    # ---------------------------------

                    elif "smoke" in class_name:

                        smoke_detected = True

                        cv2.rectangle(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            SMOKE_COLOR,
                            3
                        )

                        cv2.putText(
                            frame,
                            f"SMOKE {confidence:.0%}",
                            (x1, max(40, y1 - 15)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            SMOKE_COLOR,
                            3
                        )


        except Exception as e:

            print("Fire/smoke detection error:", e)


        # =============================================
        # DISPLAY PEOPLE COUNT
        # =============================================

        cv2.putText(
            frame,
            f"PEOPLE: {person_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            PERSON_COLOR,
            2
        )


        # =============================================
        # DISPLAY CROWD ALERT
        # =============================================

        if crowd_detected:

            cv2.putText(
                frame,
                f"CROWD ALERT: {person_count} PEOPLE",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                FIRE_COLOR,
                3
            )


        # =============================================
        # DISPLAY FALL ALERT
        # =============================================

        if fall_detected:

            cv2.putText(
                frame,
                "FALL DETECTED!",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                FIRE_COLOR,
                3
            )

        elif fall_time > 0:

            cv2.putText(
                frame,
                f"FALL SUSPECT: {int(fall_time)} sec",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 165, 255),
                2
            )


        # =============================================
        # DISPLAY FIRE ALERT
        # =============================================

        if fire_detected:

            # Flashing border
            pulse = int(time.time() * 3) % 2

            if pulse == 0:

                cv2.rectangle(
                    frame,
                    (5, 5),
                    (
                        frame.shape[1] - 5,
                        frame.shape[0] - 5
                    ),
                    FIRE_COLOR,
                    15
                )


            cv2.putText(
                frame,
                "!!! FIRE ALERT !!!",
                (300, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                FIRE_COLOR,
                4
            )


        # =============================================
        # DISPLAY SMOKE ALERT
        # =============================================

        if smoke_detected:

            cv2.putText(
                frame,
                "!!! SMOKE DETECTED !!!",
                (300, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                SMOKE_COLOR,
                3
            )


        # =============================================
        # SHOW OBJECT SUMMARY
        # =============================================

        unique_objects = list(
            dict.fromkeys(detected_objects)
        )

        summary = (
            "OBJECTS: " +
            ", ".join(unique_objects[:6])
        )

        if len(unique_objects) > 0:

            cv2.putText(
                frame,
                summary.upper(),
                (20, frame.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )


        # =============================================
        # SYSTEM TITLE
        # =============================================

        cv2.putText(
            frame,
            "SENTINE AI SAFETY SYSTEM",
            (20, frame.shape[0] - 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )


        # =============================================
        # SHOW OUTPUT
        # =============================================

        cv2.imshow(
            "Sentine AI - Safety Monitoring",
            frame
        )


        # =============================================
        # QUIT
        # =============================================

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            print("Closing system...")
            break


    close_camera(cap)


# =====================================================
# GET MODEL CLASS NAME
# =====================================================

def model_name(result, class_id):

    try:

        return result.names[class_id]

    except:

        return "unknown"


# =====================================================
# START PROGRAM
# =====================================================

if __name__ == "__main__":

    main()