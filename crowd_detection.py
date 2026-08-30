# crowd_detection.py

def check_crowd(person_count, threshold=8):
    """
    Returns True when the number of detected people
    reaches or exceeds the crowd threshold.
    """

    try:
        person_count = int(person_count)

        if person_count >= threshold:
            return True

        return False

    except (ValueError, TypeError):
        return False