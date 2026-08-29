import cv2
import time

from device import open_camera, read_frame, close_camera
from detection import detect_people
from fall_detection import detect_fall
from fire_detection import detect_fire_smoke
from crowd_detection import check_crowd
from object_detection import detect_objects


def main():

    print("=================================")
    print("Starting AI Safety System...")
    print("=================================")

    cap = open_camera(0)

    if cap is None:
        print("ERROR: Camera could not start.")
        return

    fire_alert_start = None

    while True:

       
        frame = read_frame(cap)

        if frame is None:
            print("Could not read camera frame.")
            break
        
   


        person_count = 0

        try:

            person_results = detect_people(frame)

            for result in person_results:

                if result.boxes is None:
                    continue

                for box in result.boxes:

                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0].tolist()
                    )

                    confidence = float(
                        box.conf[0]
                    )

                    person_count += 1

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"PERSON {confidence:.0%}",
                        (x1, max(30, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

        except Exception as e:
            print("Person detection error:", e)



        cv2.putText(
            frame,
            f"People: {person_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )



        crowd_detected = check_crowd(person_count)

        if crowd_detected:

            cv2.putText(
                frame,
                "CROWD ALERT",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3
            )


     

        try:

            fall_detected, fall_time = detect_fall(frame)

            if fall_time > 0:

                cv2.putText(
                    frame,
                    f"FALL SUSPECT: {int(fall_time)} sec",
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 165, 255),
                    2
                )

            if fall_detected:

                cv2.putText(
                    frame,
                    "FALL CONFIRMED!",
                    (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )

        except Exception as e:
            print("Fall detection error:", e)




        try:

            fire_results = detect_fire_smoke(frame)

            fire_detected = False
            smoke_detected = False

            for result in fire_results:

                if result.boxes is None:
                    continue

                for box in result.boxes:

                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0].tolist()
                    )

                    confidence = float(
                        box.conf[0]
                    )

                    class_id = int(
                        box.cls[0]
                    )

                    class_name = fire_model_name(
                        result,
                        class_id
                    )

                    class_name = class_name.lower()


                    # FIRE
                    if "fire" in class_name:

                        fire_detected = True

                        cv2.rectangle(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            (0, 0, 255),
                            4
                        )

                        cv2.putText(
                            frame,
                            f"FIRE! {confidence:.0%}",
                            (x1, max(40, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255),
                            3
                        )


                    # SMOKE
                    elif "smoke" in class_name:

                        smoke_detected = True

                        cv2.rectangle(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            (128, 128, 128),
                            3
                        )

                        cv2.putText(
                            frame,
                            f"SMOKE! {confidence:.0%}",
                            (x1, max(40, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (200, 200, 200),
                            2
                        )



            if fire_detected:

                if fire_alert_start is None:
                    fire_alert_start = time.time()

                # Pulsing / flashing effect
                pulse = int(
                    abs(
                        255 * (
                            time.time() % 1
                        )
                    )
                )

                overlay = frame.copy()

                cv2.rectangle(
                    overlay,
                    (0, 0),
                    (frame.shape[1], frame.shape[0]),
                    (0, 0, 255),
                    10
                )

                frame = cv2.addWeighted(
                    overlay,
                    0.5,
                    frame,
                    0.5,
                    0
                )

                cv2.putText(
                    frame,
                    "!!! FIRE ALERT !!!",
                    (
                        int(frame.shape[1] / 2) - 200,
                        100
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 0, 255),
                    5
                )


            if smoke_detected:

                cv2.putText(
                    frame,
                    "!!! SMOKE ALERT !!!",
                    (20, 200),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (100, 100, 255),
                    3
                )


        except Exception as e:

            print("Fire detection error:", e)


 

        cv2.imshow(
            "AI SAFETY SYSTEM",
            frame
        )


     
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("Closing system...")
            break


    close_camera(cap)


def fire_model_name(result, class_id):

    try:

        return result.names[class_id]

    except:

        return "unknown"


if __name__ == "__main__":

    main()