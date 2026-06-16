import pyrealsense2 as rs
import numpy as np
import cv2
import os
from datetime import datetime

# ── 1. Inställningar för sparande ────────────────────────────────────────────
# Skapa en mapp på skrivbordet där bilderna sparas
save_folder = os.path.join(os.path.expanduser("~"), "Desktop", "UR5_Batteri_Dataset")
os.makedirs(save_folder, exist_ok=True)

image_counter = 0
print(f"Bilder kommer sparas i: {save_folder}")

# ── 2. Konfigurera RealSense-kameran ─────────────────────────────────────────
pipeline = rs.pipeline()
config = rs.config()

# Vi använder 1280x720 (Standard HD) för att få knivskarpa råbilder.
# Roboflow kommer senare skala detta till 832x832 åt oss.
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

print("Startar RealSense-kameran...")
pipeline.start(config)

# För att kameran ska hinna ställa in auto-exponering och vitbalans
# väntar vi ut de första bildrutorna
for _ in range(15):
    pipeline.wait_for_frames()

try:
    print("\n--- REDO ATT FOTA ---")
    print("[MELLANSLAG] = Ta bild")
    print("[ Q ]        = Avsluta")

    while True:
        # Hämta bildruta
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Konvertera till Numpy/OpenCV-format
        color_image = np.asanyarray(color_frame.get_data())
        
        # Vi gör en kopia för skärmen så att vi kan rita instruktioner på den, 
        # men vi sparar 'color_image' (den rena bilden utan text) på hårddisken.
        display_image = color_image.copy()

        # Rita en guide-ruta i mitten som ungefär motsvarar 832x832-beskärningen 
        # (Visar ungefär vad AI:n kommer fokusera på när svarta kanter lagts till)
        h, w = display_image.shape[:2]
        center_x, center_y = w // 2, h // 2
        # Rita två diskreta linjer som visar den centrala fyrkanten
        box_size = h # Eftersom 720 är höjden, blir den beskurna boxen 720x720 i mitten
        cv2.rectangle(display_image, 
                      (center_x - box_size//2, 0), 
                      (center_x + box_size//2, h), 
                      (255, 255, 255), 1)

        # Lägg in en liten text-overlay
        cv2.putText(display_image, f"Bilder sparade: {image_counter}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_image, "Tryck MELLANSLAG for bild | Q for avslut", (20, h - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        # Visa bilden
        cv2.imshow("UR5 Datainsamling (RealSense)", display_image)

        # Läs av tangentbordet
        key = cv2.waitKey(1)
        
        if key == 32:  # 32 är ASCII-koden för Mellanslag
            # Skapa ett unikt filnamn med tidsstämpel
            timestamp = datetime.now().strftime("%H%M%S_%f")[:10]
            filename = os.path.join(save_folder, f"batteri_{timestamp}.jpg")
            
            # Spara RAW-bilden (utan utritade rutor och text)
            # Parametern [int(cv2.IMWRITE_JPEG_QUALITY), 100] ser till att vi inte tappar kvalitet
            cv2.imwrite(filename, color_image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            
            image_counter += 1
            print(f"[{image_counter}] Sparade: {filename}")
            
            # Gör en liten grön "blixt" på skärmen som visuell feedback
            flash_img = np.full_like(display_image, (0, 255, 0))
            cv2.imshow("UR5 Datainsamling (RealSense)", flash_img)
            cv2.waitKey(50) # Håll blixten i 50 millisekunder

        elif key & 0xFF == ord('q'):
            print("Avslutar insamlingen.")
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()