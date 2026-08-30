import cv2
import time
from datetime import datetime

from device import open_camera, read_frame, close_camera
from object_detection import detect_objects
from crowd_detection import detect_crowd_from_people
from fall_detection import detect_fall
from fire_detection import detect_fire_smoke





PERSON_COLOR = (0, 255, 0)
OBJECT_COLOR = (255, 170, 0)
FIRE_COLOR = (0, 0, 255)
SMOKE_COLOR = (160, 160, 160)

ORANGE = (0, 165, 255)
CYAN = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
DARK_BG = (25, 25, 25)
PANEL_COLOR = (45, 45, 45)
GREEN = (0, 255, 0)
RED = (0, 0, 255)



MIN_CONFIDENCE = 0.45
PERSON_CONFIDENCE = 0.55
FIRE_SMOKE_CONFIDENCE = 0.50

# Duplicate detection removal
DUPLICATE_IOU_THRESHOLD = 0.45

# Ignore tiny false detections
MIN_BOX_AREA = 1500

# Crowd threshold
CROWD_THRESHOLD = 5



def model_name(result, class_id):

    try:
        return result.names[class_id]

    except Exception:
        return "unknown"



def calculate_iou(box_a, box_b):

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)

    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)

    intersection = inter_width * inter_height

    area_a = max(
        1,
        (ax2 - ax1) * (ay2 - ay1)
    )

    area_b = max(
        1,
        (bx2 - bx1) * (by2 - by1)
    )

    union = (
        area_a
        + area_b
        - intersection
    )

    if union <= 0:
        return 0.0

    return intersection / union



def remove_duplicate_boxes(detections):

    if not detections:
        return []

    # Keep strongest confidence first
    detections = sorted(
        detections,
        key=lambda item: item["confidence"],
        reverse=True
    )

    final_detections = []

    for detection in detections:

        is_duplicate = False

        for existing in final_detections:

            # Compare only the same class
            if (
                detection["class_name"]
                != existing["class_name"]
            ):
                continue

            iou = calculate_iou(
                detection["box"],
                existing["box"]
            )

            # Remove heavily overlapping duplicates
            if iou >= DUPLICATE_IOU_THRESHOLD:

                is_duplicate = True
                break

        if not is_duplicate:

            final_detections.append(
                detection
            )

    return final_detections



def draw_panel(
    frame,
    x1,
    y1,
    x2,
    y2,
    color=DARK_BG,
    alpha=0.80
):

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (x1, y1),
        (x2, y2),
        color,
        -1
    )

    cv2.addWeighted(
        overlay,
        alpha,
        frame,
        1 - alpha,
        0,
        frame
    )



def draw_label(
    frame,
    text,
    x,
    y,
    text_color=WHITE,
    background=BLACK,
    scale=0.50,
    thickness=1
):

    (
        text_width,
        text_height
    ), _ = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        thickness
    )

    cv2.rectangle(
        frame,
        (x - 4, y - text_height - 8),
        (x + text_width + 6, y + 5),
        background,
        -1
    )

    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        text_color,
        thickness,
        cv2.LINE_AA
    )



def process_object_detection(frame):

    detections = []

    try:

        object_results = detect_objects(
            frame
        )

        
        for result in object_results:

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

                class_name = model_name(
                    result,
                    class_id
                ).lower()

               
                if confidence < MIN_CONFIDENCE:
                    continue

                # Stronger filtering for people
                if (
                    class_name == "person"
                    and confidence < PERSON_CONFIDENCE
                ):
                    continue

               
                box_width = x2 - x1
                box_height = y2 - y1

                if (
                    box_width <= 0
                    or box_height <= 0
                ):
                    continue

                box_area = (
                    box_width
                    * box_height
                )

                if box_area < MIN_BOX_AREA:
                    continue

                # Save detection
                detections.append({
                    "box": (
                        x1,
                        y1,
                        x2,
                        y2
                    ),
                    "confidence": confidence,
                    "class_name": class_name
                })


       
        detections = remove_duplicate_boxes(
            detections
        )


       
        person_count = 0
        detected_objects = []

        for detection in detections:

            x1, y1, x2, y2 = (
                detection["box"]
            )

            confidence = (
                detection["confidence"]
            )

            class_name = (
                detection["class_name"]
            )

            detected_objects.append(
                class_name
            )

           
            if class_name == "person":

                person_count += 1

                color = PERSON_COLOR

            else:

                color = OBJECT_COLOR


          
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )


          
            label = (
                f"{class_name.upper()} "
                f"{confidence:.0%}"
            )

            draw_label(
                frame,
                label,
                x1,
                max(25, y1 - 8),
                BLACK,
                color,
                0.50,
                1
            )


        return (
            person_count,
            detected_objects,
            detections
        )


    except Exception as error:

        print(
            "Object detection error:",
            error
        )

        return 0, [], []



def main():

    print("=" * 60)
    print("SAFE VISION")
    print("Powered by Etrosys")
    print("AI SAFETY MONITORING SYSTEM")
    print("=" * 60)


   
    cap = open_camera(0)

    if cap is None:

        print(
            "ERROR: Could not open camera."
        )

        return


    print(
        "Camera started successfully!"
    )

    print(
        "Press Q to close Safe Vision."
    )


   
    window_name = (
        "Safe Vision - AI Safety Monitoring"
    )

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL
    )

    cv2.resizeWindow(
        window_name,
        1280,
        720
    )


   
    previous_time = time.time()


    
    while True:


        
        frame = read_frame(cap)

        if frame is None:

            print(
                "ERROR: Could not read frame."
            )

            break


       
        frame = cv2.resize(
            frame,
            (1280, 720)
        )

        height, width = frame.shape[:2]


      
        current_time = time.time()

        time_difference = (
            current_time
            - previous_time
        )

        if time_difference > 0:

            fps = (
                1 / time_difference
            )

        else:

            fps = 0

        previous_time = current_time


       
        (
            person_count,
            detected_objects,
            final_detections
        ) = process_object_detection(
            frame
        )


       
        crowd_detected = False

        try:

            crowd_detected = (
                detect_crowd_from_people(
                    person_count,
                    crowd_threshold=CROWD_THRESHOLD
                )
            )

        except Exception as error:

            print(
                "Crowd detection error:",
                error
            )


      
        fall_detected = False
        fall_time = 0

        try:

            fall_result = detect_fall(
                frame
            )

            if isinstance(
                fall_result,
                tuple
            ):

                fall_detected = (
                    bool(fall_result[0])
                )

                if len(fall_result) > 1:

                    fall_time = (
                        fall_result[1]
                    )


            elif isinstance(
                fall_result,
                dict
            ):

                fall_detected = (
                    fall_result.get(
                        "fall_detected",
                        False
                    )
                )

                fall_time = (
                    fall_result.get(
                        "fall_time",
                        0
                    )
                )


            else:

                fall_detected = bool(
                    fall_result
                )


        except Exception as error:

            print(
                "Fall detection error:",
                error
            )


       
        fire_detected = False
        smoke_detected = False

        try:

            fire_results = (
                detect_fire_smoke(
                    frame
                )
            )

            fire_detections = []


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

                    if (
                        confidence
                        < FIRE_SMOKE_CONFIDENCE
                    ):
                        continue

                    class_id = int(
                        box.cls[0]
                    )

                    class_name = (
                        model_name(
                            result,
                            class_id
                        ).lower()
                    )

                    fire_detections.append({

                        "box": (
                            x1,
                            y1,
                            x2,
                            y2
                        ),

                        "confidence":
                            confidence,

                        "class_name":
                            class_name
                    })


      

            fire_detections = (
                remove_duplicate_boxes(
                    fire_detections
                )
            )


        

            for detection in (
                fire_detections
            ):

                (
                    x1,
                    y1,
                    x2,
                    y2
                ) = detection["box"]

                confidence = (
                    detection["confidence"]
                )

                class_name = (
                    detection["class_name"]
                )


                # FIRE

                if "fire" in class_name:

                    fire_detected = True

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        FIRE_COLOR,
                        3
                    )

                    draw_label(
                        frame,
                        f"FIRE {confidence:.0%}",
                        x1,
                        max(30, y1 - 10),
                        WHITE,
                        FIRE_COLOR,
                        0.60,
                        2
                    )


                # SMOKE

                elif "smoke" in class_name:

                    smoke_detected = True

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        SMOKE_COLOR,
                        3
                    )

                    draw_label(
                        frame,
                        f"SMOKE {confidence:.0%}",
                        x1,
                        max(30, y1 - 10),
                        BLACK,
                        SMOKE_COLOR,
                        0.60,
                        2
                    )


        except Exception as error:

            print(
                "Fire/smoke detection error:",
                error
            )


      
        threat_level = "SAFE"
        threat_color = GREEN

        if crowd_detected:

            threat_level = "CAUTION"
            threat_color = ORANGE

        if (
            fall_detected
            or smoke_detected
        ):

            threat_level = "WARNING"
            threat_color = ORANGE

        if fire_detected:

            threat_level = "CRITICAL"
            threat_color = RED


       

        draw_panel(
            frame,
            0,
            0,
            width,
            80,
            DARK_BG,
            0.88
        )


        # SAFE VISION TITLE

        cv2.putText(
            frame,
            "SAFE VISION",
            (25, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.95,
            CYAN,
            2,
            cv2.LINE_AA
        )


        # POWERED BY ETROSYS

        cv2.putText(
            frame,
            "Powered by Etrosys",
            (27, 63),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            WHITE,
            1,
            cv2.LINE_AA
        )


        # LIVE MONITORING

        live_pulse = (
            int(
                time.time() * 2
            ) % 2
        )

        if live_pulse == 0:

            live_color = RED

        else:

            live_color = (
                0,
                120,
                255
            )


        cv2.circle(
            frame,
            (width - 220, 30),
            7,
            live_color,
            -1
        )


        cv2.putText(
            frame,
            "LIVE MONITORING",
            (width - 200, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            WHITE,
            2,
            cv2.LINE_AA
        )


        # DATE AND TIME

        current_datetime = (
            datetime.now().strftime(
                "%d-%m-%Y  %H:%M:%S"
            )
        )

        cv2.putText(
            frame,
            current_datetime,
            (width - 250, 64),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (180, 180, 180),
            1,
            cv2.LINE_AA
        )


       

        panel_x1 = 15
        panel_y1 = 100
        panel_x2 = 280
        panel_y2 = 350

        draw_panel(
            frame,
            panel_x1,
            panel_y1,
            panel_x2,
            panel_y2,
            PANEL_COLOR,
            0.82
        )

        cv2.rectangle(
            frame,
            (
                panel_x1,
                panel_y1
            ),
            (
                panel_x2,
                panel_y2
            ),
            CYAN,
            1
        )


        # PANEL TITLE

        cv2.putText(
            frame,
            "SYSTEM STATUS",
            (30, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            CYAN,
            2,
            cv2.LINE_AA
        )


        # FPS

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (30, 165),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            WHITE,
            1
        )


        # PEOPLE COUNT

        cv2.putText(
            frame,
            f"PEOPLE: {person_count}",
            (30, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            PERSON_COLOR,
            2
        )


        # CAMERA STATUS

        cv2.putText(
            frame,
            "CAMERA: ONLINE",
            (30, 235),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            GREEN,
            1
        )


        # AI ENGINE STATUS

        cv2.putText(
            frame,
            "AI ENGINE: ACTIVE",
            (30, 265),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            GREEN,
            1
        )


        # THREAT LEVEL

        cv2.putText(
            frame,
            "THREAT LEVEL:",
            (30, 300),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            WHITE,
            1
        )

        cv2.putText(
            frame,
            threat_level,
            (30, 330),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            threat_color,
            2
        )


      
        has_alert = (
            crowd_detected
            or fall_detected
            or smoke_detected
            or fire_detected
        )


        if has_alert:

            alert_x1 = 15
            alert_y1 = 380
            alert_x2 = 390
            alert_y2 = 590

            draw_panel(
                frame,
                alert_x1,
                alert_y1,
                alert_x2,
                alert_y2,
                DARK_BG,
                0.85
            )

            cv2.rectangle(
                frame,
                (
                    alert_x1,
                    alert_y1
                ),
                (
                    alert_x2,
                    alert_y2
                ),
                RED,
                1
            )

            cv2.putText(
                frame,
                "ACTIVE ALERTS",
                (30, 415),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                RED,
                2
            )

            current_alert_y = 455


            # CROWD

            if crowd_detected:

                cv2.putText(
                    frame,
                    f"! CROWD: {person_count} PEOPLE",
                    (
                        30,
                        current_alert_y
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    ORANGE,
                    2
                )

                current_alert_y += 35


            # FALL

            if fall_detected:

                cv2.putText(
                    frame,
                    "! FALL DETECTED",
                    (
                        30,
                        current_alert_y
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    RED,
                    2
                )

                current_alert_y += 35


            elif fall_time > 0:

                cv2.putText(
                    frame,
                    (
                        f"? FALL SUSPECT: "
                        f"{int(fall_time)} SEC"
                    ),
                    (
                        30,
                        current_alert_y
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    ORANGE,
                    2
                )

                current_alert_y += 35


            # SMOKE

            if smoke_detected:

                cv2.putText(
                    frame,
                    "! SMOKE DETECTED",
                    (
                        30,
                        current_alert_y
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    SMOKE_COLOR,
                    2
                )

                current_alert_y += 35


            # FIRE

            if fire_detected:

                cv2.putText(
                    frame,
                    "!!! FIRE DETECTED !!!",
                    (
                        30,
                        current_alert_y
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    RED,
                    2
                )


      

        if fire_detected:

            pulse = (
                int(
                    time.time() * 3
                ) % 2
            )

            if pulse == 0:

                cv2.rectangle(
                    frame,
                    (5, 5),
                    (
                        width - 5,
                        height - 5
                    ),
                    FIRE_COLOR,
                    8
                )


       

        unique_objects = list(
            dict.fromkeys(
                detected_objects
            )
        )

        if unique_objects:

            object_summary = (
                ", ".join(
                    unique_objects[:6]
                )
            )

        else:

            object_summary = (
                "NO SIGNIFICANT OBJECTS"
            )


        draw_panel(
            frame,
            0,
            height - 45,
            width,
            height,
            DARK_BG,
            0.88
        )


        cv2.putText(
            frame,
            "DETECTED:",
            (20, height - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            CYAN,
            1
        )

        cv2.putText(
            frame,
            object_summary.upper(),
            (115, height - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            WHITE,
            1
        )


       
        cv2.imshow(
            window_name,
            frame
        )


       

        key = (
            cv2.waitKey(1)
            & 0xFF
        )

        if key == ord("q"):

            print(
                "Closing Safe Vision..."
            )

            break


    

    close_camera(cap)

    cv2.destroyAllWindows()




if __name__ == "__main__":

    main()