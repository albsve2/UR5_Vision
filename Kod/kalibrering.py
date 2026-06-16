import time
import numpy as np
import cv2
import pyrealsense2 as rs
import rtde_control
import rtde_receive

# ── 1. Inställningar ─────────────────────────────────────────────────────────
ROBOT_IP      = "130.238.198.10" 
SCAN_POSE = [0.29174, -0.01730, 0.13128, 3.0854, 0.1221, -0.0040] 

WIDTH, HEIGHT = 1280, 720
N_POINTS      = 9          
VELOCITY      = 0.5        
ACCEL         = 0.3
OUT_FILE      = "hand_eye_affine_9.npy"

HINTS = ["I mitten", "Uppe till vänster", "Uppe till höger", "Nere till höger", "Nere till vänster", "Mitten uppe", "Mitten nere", "mitten höger", "mitten vänster"]

# ── 2. Funktioner ────────────────────────────────────────────────────────────
def start_camera():
    print("Startar RealSense...")
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, 30)
    pipe.start(cfg)
    for _ in range(30):
        pipe.wait_for_frames()          
    return pipe

def grab_frame(pipe):
    frames = pipe.wait_for_frames()
    return np.asanyarray(frames.get_color_frame().get_data())

def click_pixel(pipe, hint):
    state = {"pt": None}
    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["pt"] = (x, y)

    win = "Kalibrering"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    
    while True:
        img = grab_frame(pipe)
        cv2.putText(img, f"1. Lagg batteriet: {hint}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(img, "2. Klicka exakt i mitten pa batteriet och tryck ENTER", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if state["pt"]:
            x, y = state["pt"]
            cv2.line(img, (x - 15, y), (x + 15, y), (0, 0, 255), 2)
            cv2.line(img, (x, y - 15), (x, y + 15), (0, 0, 255), 2)
            
        cv2.imshow(win, img)
        key = cv2.waitKey(30) & 0xFF
        if key == 13 and state["pt"]:       
            cv2.destroyWindow(win)
            return state["pt"]
        if key == 27:                       
            cv2.destroyWindow(win)
            return None

# ── 3. Huvudprogram ──────────────────────────────────────────────────────────
def main():
    pipe = start_camera()
    print(f"Ansluter till UR5 på {ROBOT_IP}...")
    
    # Receive-interfacet kraschar inte av att vi joggar, så det kan vara igång hela tiden.
    try:
        rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
    except Exception as e:
        print(f"Kunde inte ansluta mottagaren: {e}")
        return

    pixels, robot_xy = [], []
    
    try:
        # 1. Vi ansluter kontrollen en första gång för att åka till start
        rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
        print("\nÅker till din förbestämda skanningsposition...")
        scan_joints = rtde_c.getInverseKinematics(SCAN_POSE)
        rtde_c.moveJ(scan_joints, VELOCITY, ACCEL)
        
        for i in range(N_POINTS):
            print(f"\n{'='*40}\n── PUNKT {i + 1} AV {N_POINTS} ──")
            
            # Steg 1: Klicka i bilden
            pt = click_pixel(pipe, HINTS[i % len(HINTS)])
            if pt is None:
                print("Kalibrering avbruten.")
                return
            
            # MAGIN: Vi stänger av Python-kontrollen med flit!
            rtde_c.disconnect()
            time.sleep(0.5)

            # Steg 2: Användaren joggar fritt med skärmen
            print(" -> Robotens lås är släppt! Jogga klon till batteriet med pekskärmen.")
            print(" -> Tryck ENTER här i terminalen när klon är exakt över batteriet...")
            input()
            
            # Läs av den fysiska positionen (rtde_r är fortfarande vaken)
            pose = rtde_r.getActualTCPPose()
            xy = (pose[0], pose[1])
            
            pixels.append(pt)
            robot_xy.append(xy)
            print(f"   Sparat! Pixel {pt} -> Robot ({xy[0]:.4f}, {xy[1]:.4f})")

            # Steg 3: Koppla upp igen och åk tillbaka!
            print(" -> Återansluter till roboten och åker tillbaka till skanningspositionen...")
            rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
            rtde_c.moveJ(scan_joints, VELOCITY, ACCEL)

        print("\nBeräknar transformationsmatris...")
        src = np.array(pixels, dtype=np.float32)
        dst = np.array(robot_xy, dtype=np.float32)
        M, _ = cv2.estimateAffine2D(src, dst)
        
        if M is None:
            print("FEL: Kunde inte räkna ut matrisen.")
            return

        proj = (M @ np.vstack([src.T, np.ones(len(src))])).T
        err_mm = np.linalg.norm(proj - dst, axis=1) * 1000.0
        
        print("\nKLAR! Kalibreringsmatris:\n", M)
        print(f"\nMedelfel: {err_mm.mean():.2f} mm | Max-fel: {err_mm.max():.2f} mm")
        
        np.save(OUT_FILE, M)
        print(f"SUCCÉ! Matrisen sparades till: {OUT_FILE}")

    except Exception as e:
        print(f"\nAnnat fel uppstod: {e}")
    finally:
        pipe.stop()
        try:
            rtde_c.disconnect()
        except:
            pass
        rtde_r.disconnect()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()