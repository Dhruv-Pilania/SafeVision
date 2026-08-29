def detect_crowd_from_people(person_count, crowd_threshold=7):

    if person_count >= crowd_threshold:
        return True

    return False