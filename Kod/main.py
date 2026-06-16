"""
Pick-and-place: UR5 (CB3, Polyscope 3.5.1) + RealSense + YOLO26 + OnRobot RG2

Innehåller:
 - Optimerad Dashboard-överlämning för att undvika RG2-kraschar.
 - Fix för 200mm TCP-offset via setTcp().
 - Stabil vinkelberäkning med Image Moments (inget 90-gradershopp).
 - Kamerafönster utan text/statusbar för rent flöde.
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import math
import time
import threading
import socket
from ultralytics import YOLO
import rtde_control 
import rtde_receive

# ==============================================================================
# KAMERA
# ==============================================================================
print("Letar efter kamera...")
ctx = rs.context()
if len(ctx.query_devices()) == 0:
    print("Ingen RealSense hittad."); exit()
for d in ctx.query_devices():
    print("   ", d.get_info(rs.camera_info.name),
          "| USB:", d.get_info(rs.camera_info.usb_type_descriptor))

pipeline = rs.pipeline()
config   = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
try:
    pipeline.start(config)
except Exception as e:
    print(f"Kunde inte starta kameran: {e}"); exit()

print("Värmer upp kameran...")
for _ in range(80):
    if pipeline.poll_for_frames().size() > 0:
        print("Kameran redo!"); break
    time.sleep(0.05)

# ==============================================================================
# MODELL OCH KALIBRERING
# ==============================================================================
MODEL_PATH = "runs/segment/runs/segment/yolo26m_seg_batterier-2/weights/best.pt"
print("Startar YOLO-26 modell...")
model = YOLO(MODEL_PATH, task="segment")
print("Värmer upp modellen...")
model(np.zeros((720, 1280, 3), dtype=np.uint8), conf=0.75, verbose=False)
print("Modell redo!")

try:
    M = np.load("hand_eye_affine_9.npy")
    print("Kalibreringsmatris laddad!")
except FileNotFoundError:
    print("Hittade inte 'hand_eye_affine_9.npy'."); exit()

# ==============================================================================
# INSTÄLLNINGAR
# ==============================================================================
ROBOT_IP = "130.238.198.10"

SCAN_POSE = [0.29174, -0.01730, 0.13128, 3.0854, 0.1221, -0.0040]
DROP_POSE = [0.42365, 0.23865, 0.13128, 3.0854, 0.1221, -0.0040]
PICK_Z    = 0.030

# Tvingar python att fatta var gripperns spets är (147.6 mm offset, 90 deg vinkel)
TCP_OFFSET = [-0.1476, 0.0, 0.050, 0.0, -1.571, 0.0]

ANGLE_OFFSET = 0
VELOCITY_FAST, ACCEL_FAST = 1.0, 1.0 #0.15, 0.1
VELOCITY_SLOW, ACCEL_SLOW = 0.25, 0.25 #0.05, 0.05

GRIP_CLOSE_PROG = "grip_close.urp"   # program i /programs/ på roboten
GRIP_OPEN_PROG  = "grip_open.urp"
GRIP_WAIT       = 1.5   # Optimerad väntetid för OnRobot-mekaniken

# ==============================================================================
# DASHBOARD-KLIENT  (port 29999)
# ==============================================================================
class Dashboard:
    def __init__(self, ip, port=29999):
        self._s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._s.settimeout(5.0)
        self._s.connect((ip, port))
        self._s.recv(1024)          # läs välkomstmeddelandet

    def _cmd(self, text):
        self._s.send((text + "\n").encode())
        try:
            resp = self._s.recv(1024).decode().strip()
        except socket.timeout:
            resp = ""
        print(f"   dash › {text!r} → {resp!r}")
        return resp

    def load(self, program):        # ladda program från /programs/
        return self._cmd(f"load {program}")

    def play(self):
        return self._cmd("play")

    def stop(self):
        return self._cmd("stop")

    def close(self):
        try: self._cmd("quit")
        except Exception: pass
        self._s.close()

# ==============================================================================
# DELAT TILLSTÅND
# ==============================================================================
lock    = threading.Lock()
running = threading.Event()
running.set()
shared  = {"frame": None, "detection": None}

# ==============================================================================
# FUNKTIONER
# ==============================================================================
def add_z_rotation(rx, ry, rz, angle_degrees):
    R_original, _ = cv2.Rodrigues(np.array([rx, ry, rz], dtype=np.float64))
    theta = math.radians(angle_degrees)
    R_z   = np.array([
        [math.cos(theta), -math.sin(theta), 0],
        [math.sin(theta),  math.cos(theta), 0],
        [0,                0,               1]
    ])
    rvec, _ = cv2.Rodrigues(R_z @ R_original)
    return rvec.flatten().tolist()

def pixel_to_robot(pixel_x, pixel_y, stable_angle_deg):
    robot_xy = M @ np.array([pixel_x, pixel_y, 1.0])
    print(f"\nPixel: ({pixel_x:.1f}, {pixel_y:.1f})  →  "
          f"Robot: ({robot_xy[0]:.4f}, {robot_xy[1]:.4f})\n")
    
    final_angle = - ((stable_angle_deg + ANGLE_OFFSET) % 180 - 90)
    
    rx, ry, rz  = add_z_rotation(SCAN_POSE[3], SCAN_POSE[4], SCAN_POSE[5], final_angle)
    
    return ([robot_xy[0], robot_xy[1], SCAN_POSE[2], rx, ry, rz],
            [robot_xy[0], robot_xy[1], PICK_Z,        rx, ry, rz])

def wait_move(local_c, move_fn, *args, timeout=20.0):
    move_fn(*args, asynchronous=True)
    t0 = time.time()
    while time.time() - t0 < 0.2:
        if not local_c.isSteady():
            break
        time.sleep(0.02)
    t0 = time.time()
    while running.is_set() and (time.time() - t0) < timeout:
        if local_c.isSteady():
            break
        time.sleep(0.04)

def actuate_gripper(local_c, dash, action):
    program = GRIP_CLOSE_PROG if action == "close" else GRIP_OPEN_PROG
    print(f" Gripper {action} via Dashboard...")

    local_c.disconnect()
    time.sleep(0.15) 

    dash.load(program)
    time.sleep(0.05) 
    dash.play()
    time.sleep(GRIP_WAIT) 

    dash.stop()
    time.sleep(0.15) 

    new_c = rtde_control.RTDEControlInterface(ROBOT_IP)
    new_c.setTcp(TCP_OFFSET) # <-- Lösningen på 200mm felet vid återanslutning
    return new_c

# ==============================================================================
# ROBOTTRÅD
# ==============================================================================
def robot_worker():
    print(f"⏳ Ansluter till UR5 och Dashboard på {ROBOT_IP}...")
    try:
        local_c = rtde_control.RTDEControlInterface(ROBOT_IP)
        local_c.setTcp(TCP_OFFSET) # <-- Lösningen på 200mm felet vid start
        local_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
        dash    = Dashboard(ROBOT_IP)
    except Exception as e:
        print(f"❌ Kunde inte ansluta: {e}")
        running.clear(); return

    print("Robot och Dashboard anslutna.")
    scan_joints = local_c.getInverseKinematics(SCAN_POSE)

    while running.is_set():
        wait_move(local_c, local_c.moveJ, scan_joints, VELOCITY_FAST, ACCEL_FAST)
        time.sleep(0.15) 
        if not running.is_set():
            break

        with lock:
            frame = shared["frame"]
        if frame is None:
            time.sleep(0.05); continue

        results = model(frame, conf=0.75, verbose=False)

        if results[0].masks is not None and len(results[0].masks.xy) > 0:
            mask_pts = np.array(results[0].masks.xy[0], dtype=np.int32)
            
            # --- STABIL VINKELBERÄKNING (IMAGE MOMENTS) ---
            M_moments = cv2.moments(mask_pts)
            if M_moments['m00'] != 0:
                c_x = M_moments['m10'] / M_moments['m00']
                c_y = M_moments['m01'] / M_moments['m00']
                
                angle_rad = 0.5 * math.atan2(2 * M_moments['mu11'], M_moments['mu20'] - M_moments['mu02'])
                stable_angle_deg = math.degrees(angle_rad)
                
                with lock:
                    shared["detection"] = {"center": (int(c_x), int(c_y)), "mask": mask_pts}

                hover_pose, pick_pose = pixel_to_robot(c_x, c_y, stable_angle_deg)
                time.sleep(0.4)
                
                with lock: shared["detection"] = None
                if not running.is_set(): break

                # Transport & Grip
                wait_move(local_c, local_c.moveJ_IK, hover_pose, VELOCITY_FAST, ACCEL_FAST)
                wait_move(local_c, local_c.moveL,    pick_pose,  VELOCITY_SLOW, ACCEL_SLOW)
                time.sleep(0.3)

                local_c = actuate_gripper(local_c, dash, "close")

                wait_move(local_c, local_c.moveL,    hover_pose, VELOCITY_FAST, ACCEL_FAST)
                wait_move(local_c, local_c.moveJ_IK, DROP_POSE,  VELOCITY_FAST, ACCEL_FAST)

                local_c = actuate_gripper(local_c, dash, "open")

        else:
            with lock:
                shared["detection"] = None
            time.sleep(0.3)

    try:
        local_c.stopScript(); local_c.disconnect()
        local_r.disconnect(); dash.close()
    except Exception:
        pass

# ==============================================================================
# KAMERA + FÖNSTER
# ==============================================================================
cv2.namedWindow("Live Kamera", cv2.WINDOW_AUTOSIZE)
worker = threading.Thread(target=robot_worker, daemon=True)
worker.start()

print("\n STARTAR. Tryck ESC i kamerafönstret för att avbryta.")
disp = np.zeros((720, 1280, 3), dtype=np.uint8)

# Ritar text för allra första bilduppstarten, men tas sedan bort när kameran matar frames
cv2.putText(disp, "Vantar pa kamera...", (40, 360),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

try:
    while running.is_set():
        fs = pipeline.poll_for_frames()
        if fs.size() > 0:
            cf = fs.get_color_frame()
            if cf:
                clean = np.asanyarray(cf.get_data()).copy()
                with lock:
                    shared["frame"] = clean
                    det    = shared["detection"]

                disp = clean.copy()
                if det is not None:
                    mask    = det["mask"]
                    overlay = disp.copy()
                    
                    # Ritar ut masken över objektet
                    cv2.fillPoly(overlay, [mask], (0, 255, 0))
                    cv2.addWeighted(overlay, 0.4, disp, 0.6, 0, disp)
                    cv2.drawContours(disp, [mask], -1, (0, 255, 0), 2)
                    cv2.circle(disp, det["center"], 5, (0, 0, 255), -1)
                    
        else:
            time.sleep(0.003)

        cv2.imshow("Live Kamera", disp)
        if (cv2.waitKey(1) & 0xFF) == 27:
            running.clear(); break

except KeyboardInterrupt:
    running.clear()

finally:
    print("\nAvslutar...")
    running.clear()
    worker.join(timeout=10)
    try: pipeline.stop()
    except Exception: pass
    cv2.destroyAllWindows()