import cv2


def open_camera(camera_index=0):
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("ERROR: Camera could not be opened.")
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Camera opened successfully!")
    return cap


def read_frame(cap):
    if cap is None:
        return None

    success, frame = cap.read()

    if not success:
        return None

    return frame


def close_camera(cap):
    if cap is not None:
        cap.release()

    cv2.destroyAllWindows()