import time
import rtde_io # <-- Vi använder IO-biblioteket istället!

ROBOT_IP = "130.238.198.10"

def test_gripper():
    print(f"Ansluter till UR5 IO-gränssnitt på {ROBOT_IP}...")
    try:
        # rtde_io tar INTE över robotens rörelser, den flippar bara switchar i bakgrunden!
        io = rtde_io.RTDEIOInterface(ROBOT_IP)
    except Exception as e:
        print(f"Kunde inte ansluta: {e}")
        return

    print("\nTestar grippern")
    time.sleep(1)

    # 1. Stäng klon
    print(" Stänger grippern...")
    io.setStandardDigitalOut(1, False) # Se till att "öppna" är av
    io.setStandardDigitalOut(0, True)  # Skicka "stäng"
    time.sleep(2.5) 

    # 2. Öppna klon
    print(" Öppnar grippern...")
    io.setStandardDigitalOut(0, False) # Se till att "stäng" är av
    io.setStandardDigitalOut(1, True)  # Skicka "öppna"
    time.sleep(2.5) 

    # 3. Återställ utgångarna
    print("Återställer signalerna...")
    io.setStandardDigitalOut(0, False)
    io.setStandardDigitalOut(1, False)

    io.disconnect()
    print("Testet är klart!")

if __name__ == "__main__":
    test_gripper()