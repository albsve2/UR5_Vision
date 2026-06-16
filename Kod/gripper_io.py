import socket


def send_move_cmd(pose):
    # Skickar en rå URScript-rörelse till port 30002
    # Detta stör INTE OnRobot-grippern på en 3.5.1-robot!
    script = (
        f"def move_to_target():\n"
        f"  movej(p{pose}, a=0.1, v=0.1)\n"
        f"end\n"
    )
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(("130.238.198.10", 30002))
    s.send(script.encode())
    s.close()