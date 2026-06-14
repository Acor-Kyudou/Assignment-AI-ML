import cv2
import math
from ultralytics import YOLO

model = YOLO('best.pt')
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

W_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H_frame = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
if W_frame == 0 or H_frame == 0:
    ret, temp_frame = cap.read()
    if ret:
        H_frame, W_frame = temp_frame.shape[:2]

x_fc, y_fc = W_frame / 2, H_frame / 2
D_max = math.sqrt((W_frame / 2)**2 + (H_frame / 2)**2)

saved_shots = 0
max_shots = 2
print(f"{'Detections':<10} | {'Persons':<10} | {'C':<5} | {'S':<5} | {'P':<5} | {'T':<5} | {'Tier'}")
print("-" * 65)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.25, iou=0.45, verbose=False)
    frame_saved_this_loop = False

    for result in results:
        boxes = result.boxes
        det_count = len(boxes)
        
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            C = float(box.conf[0])

            # Scale (S)
            w_bb = x2 - x1
            h_bb = y2 - y1
            S = (w_bb * h_bb) / (W_frame * H_frame)

            # Proximity (P)
            x_c = x1 + (w_bb / 2)
            y_c = y1 + (h_bb / 2)
            dist = math.sqrt((x_c - x_fc)**2 + (y_c - y_fc)**2)
            P = 1 - (dist / D_max)

            # Final Threat Score (T)
            T = (0.5 * C) + (0.3 * S) + (0.2 * P)

            # Logic Tiers (Using Blue BGR color schemes)
            if T >= 0.70:
                color = (255, 0, 0)       
                label_tier = "CRITICAL"
                thickness = 3
            elif T >= 0.40:
                color = (255, 150, 50)    
                label_tier = "ELEVATED"
                thickness = 2
            else:
                color = (150, 0, 0)       
                label_tier = "ROUTINE"
                thickness = 1

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
            text = f"{label_tier} | T:{T:.2f} (C:{C:.2f})"
            cv2.putText(frame, text, (int(x1), int(y1) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if T >= 0.40:
                print(f"{det_count:<10} | N/A (Live) | {C:.2f} | {S:.2f} | {P:.2f} | {T:.2f} | {label_tier}")

            if T >= 0.40 and saved_shots < max_shots and not frame_saved_this_loop:
                filename = f"live_demo_capture_{saved_shots + 1}.jpg"
                cv2.imwrite(filename, frame)
                print(f">>> SAVED: {filename} to project folder.")
                saved_shots += 1
                frame_saved_this_loop = True

    cv2.imshow('Project Guardian - Threat Engine', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()