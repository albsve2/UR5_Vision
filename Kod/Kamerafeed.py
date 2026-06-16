import pyrealsense2 as rs
import numpy as np
import cv2
import math
from ultralytics import YOLO

# ── 1. Inställningar ─────────────────────────────────────────────────────────
# Uppdaterad till din nya segmenteringsmodell!
MODEL_PATH = "runs/segment/runs/segment/yolo26m_seg_batterier-2/weights/best.pt"

CONFIDENCE_THRESHOLD = 0.60  

print(f"Laddar SEGMENTERINGS-modell från: {MODEL_PATH}...")
# Viktigt: task="segment"
model = YOLO(MODEL_PATH, task="segment")

# ── 2. Konfigurera RealSense-kameran ─────────────────────────────────────────
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

print("Startar RealSense-kameran... (Tryck på 'q' för att avsluta)")
pipeline.start(config)

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Kopiera bilden (en för inferens, en att rita på)
        color_image = np.asanyarray(color_frame.get_data())
        annotated_frame = color_image.copy()

        # ── 3. Kör detektion ──────────────────────────────────────────────────
        results = model(color_image, conf=CONFIDENCE_THRESHOLD, verbose=False)

        # ── 4. Manuell utritning av Segmenteringsmasker ───────────────────────
        if results[0].masks is not None:
            # Hämta polygonpunkterna för masken
            masks_xy = results[0].masks.xy 
            
            # Hämta klasser och säkerhet (notera att dessa ligger under .boxes för segmentering)
            classes = results[0].boxes.cls.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            
            for mask_points, cls, conf in zip(masks_xy, classes, confs):
                # Vi behöver minst 3 punkter för att skapa en form
                if len(mask_points) < 3:
                    continue
                    
                class_name = model.names[int(cls)]
                
                # Konvertera maskens punkter till OpenCV-format (int32)
                pts = np.array(mask_points, dtype=np.int32)
                
                # -- MATTEMAGIN: Beräkna gripvinkel och centrum --
                # Hittar den minsta roterade rektangeln som omsluter masken
                rect = cv2.minAreaRect(pts)
                (c_x, c_y), (w, h), angle_deg = rect
                
                # Hämta de 4 hörnen på den beräknade OpenCV-rektangeln
                box_corners = cv2.boxPoints(rect)
                box_corners = np.int32(box_corners)

                # -- Utritning --
                # 1. Rita själva batteriets form (grön mask)
                cv2.polylines(annotated_frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                
                # 2. Rita OpenCV:s beräknade grip-box (röd box)
                cv2.polylines(annotated_frame, [box_corners], isClosed=True, color=(0, 0, 255), thickness=2)
                
                # 3. Rita en prick i exakt centrum (där roboten ska greppa)
                cv2.circle(annotated_frame, (int(c_x), int(c_y)), 4, (255, 0, 0), -1)
                
                # 4. Text och bakgrund
                label = f"{class_name} {conf*100:.0f}% (Vinkel: {angle_deg:.1f})"
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                text_x, text_y = int(c_x) - 20, int(c_y) - 20
                
                cv2.rectangle(annotated_frame, (text_x, text_y - text_size[1] - 5), (text_x + text_size[0], text_y + 5), (0, 0, 255), -1)
                cv2.putText(annotated_frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # ── 5. Visa Resultatet ───────────────────────────────────────────────
        cv2.imshow('UR5 Batterisortering - Segmentering', annotated_frame)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()