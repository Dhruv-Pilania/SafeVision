import time


def send_alert(alert_type, message):

    print("\n" + "=" * 50)
    print("ALERT TYPE:", alert_type)
    print("MESSAGE:", message)
    print("TIME:", time.strftime("%H:%M:%S"))
    print("=" * 50 + "\n")


def crowd_alert(person_count):

    send_alert(
        "CROWD",
        f"Crowd detected. Person count: {person_count}"
    )


def fall_alert():

    send_alert(
        "FALL",
        "Person has fallen and remained down"
    )


def fire_alert():

    send_alert(
        "FIRE",
        "Fire or smoke detected!"
    )