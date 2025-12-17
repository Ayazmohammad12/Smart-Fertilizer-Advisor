import cv2
import numpy as np
from ultralytics import YOLO
import time
from flask import Flask, Response, render_template_string
from flask_cors import CORS

# Initialize Flask
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# ================= CONFIGURATION =================
# CHANGED: Updated to look for best.pt in your root folder
MODEL_PATH = "best.pt" 
CONF_THRESHOLD = 0.85        
NECROSIS_THRESHOLD = 0.10    
STABILITY_FRAMES = 5         
DISPLAY_DURATION = 10        

# === THE SCIENTIFIC DATABASE ===
DISEASE_DB = {
    "Tomato_Early_blight": {
        "title": "EARLY BLIGHT (Detected)",
        "cause": "Nutrient Imbalance",
        "fert": ["Balanced NPK (10-10-10)", "Epsom Salt (Magnesium)", "Vermicompost"],
        "tip": "Balanced nutrition reduces leaf stress and susceptibility."
    },
    "Tomato_Late_blight": {
        "title": "LATE BLIGHT (Fungal)",
        "cause": "Fungal Spread",
        "fert": ["Potassium-rich (5-10-10)", "Calcium Nitrate", "Potassium Sulphate"],
        "tip": "Potassium strengthens plant resistance and reduces fungal spread.",
        "avoid": "Excess Nitrogen (Urea)"
    },
    "Healthy": {
        "title": "VITAL SIGNS: HEALTHY",
        "cause": "Optimal Condition",
        "fert": ["Standard N-P-K Maintenance"],
        "tip": "Maintenance of soil health prevents future outbreaks."
    }
}

# ================= SYSTEM INITIALIZATION =================
try:
    model = YOLO(MODEL_PATH)
except:
    print("Warning: Custom weights not found. Downloading standard model for demo.")
    model = YOLO('yolov8n-cls.pt') 

# State Variables
current_state = "SCANNING"
lock_timer = 0
locked_disease = None
stability_counter = 0
last_seen_disease = None

def get_tissue_analysis(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask_brown = cv2.inRange(hsv, (10, 40, 20), (30, 255, 200))
    mask_green = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    b_pix = cv2.countNonZero(mask_brown)
    g_pix = cv2.countNonZero(mask_green)
    total = b_pix + g_pix
    return b_pix / total if total > 5000 else 0

def draw_futuristic_ui(frame, box, state, disease_key, timer_left):
    x1, y1, x2, y2 = box
    overlay = frame.copy()
    color = (0, 255, 255) if state == "SCANNING" else (0, 255, 0)
    if state == "LOCKED" and "blight" in str(disease_key).lower():
        color = (0, 0, 255)

    t, l = 3, 40
    cv2.line(frame, (x1, y1), (x1+l, y1), color, t)
    cv2.line(frame, (x1, y1), (x1, y1+l), color, t)
    cv2.line(frame, (x2, y2), (x2-l, y2), color, t)
    cv2.line(frame, (x2, y2), (x2, y2-l), color, t)

    if state == "LOCKED" and disease_key in DISEASE_DB:
        info = DISEASE_DB[disease_key]
        cv2.rectangle(overlay, (x2 + 30, y1), (x2 + 500, y2 + 120), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        cv2.putText(frame, info['title'], (x2 + 45, y1 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(frame, f"CAUSE: {info['cause']}", (x2 + 45, y1 + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, "RECOMMENDED FERTILIZERS:", (x2 + 45, y1 + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        y_off = y1 + 145
        for f in info['fert']:
            cv2.putText(frame, f"-> {f}", (x2 + 45, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_off += 25
        cv2.rectangle(frame, (x2 + 45, y_off + 10), (x2 + 480, y_off + 80), color, 1)
        cv2.putText(frame, "FIELD TIP (VIVA):", (x2 + 55, y_off + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        tip = info['tip']
        cv2.putText(frame, tip[:42], (x2 + 55, y_off + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if len(tip) > 42:
            cv2.putText(frame, tip[42:], (x2 + 55, y_off + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"RESETTING IN: {int(timer_left)}s", (x2 + 45, y2 + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

    cv2.putText(frame, f"SYSTEM: {state}", (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return frame

def generate_frames():
    global current_state, lock_timer, locked_disease, stability_counter, last_seen_disease
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        box_s = 350
        x1, y1 = (w - box_s)//2, (h - box_s)//2
        x2, y2 = x1 + box_s, y1 + box_s
        roi = frame[y1:y2, x1:x2]

        if current_state == "SCANNING":
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv_roi, (35, 40, 40), (85, 255, 255))
            if cv2.countNonZero(mask) > 8000:
                results = model(roi, verbose=False)
                name = model.names[results[0].probs.top1]
                conf = float(results[0].probs.top1conf)
                rot_ratio = get_tissue_analysis(roi)
                if "blight" in name.lower() and rot_ratio < NECROSIS_THRESHOLD:
                    name = "Healthy" 
                if conf > CONF_THRESHOLD:
                    if name == last_seen_disease:
                        stability_counter += 1
                    else:
                        stability_counter = 0
                        last_seen_disease = name
                    if stability_counter >= STABILITY_FRAMES:
                        current_state = "LOCKED"
                        locked_disease = name
                        lock_timer = time.time()
            else:
                stability_counter = 0
        elif current_state == "LOCKED":
            elapsed = time.time() - lock_timer
            if elapsed > DISPLAY_DURATION:
                current_state = "SCANNING"
                stability_counter = 0
                locked_disease = None
        
        timer_val = DISPLAY_DURATION - (time.time() - lock_timer) if current_state == "LOCKED" else 0
        frame = draw_futuristic_ui(frame, (x1, y1, x2, y2), current_state, locked_disease, timer_val)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    # CHANGED: Added host='0.0.0.0' for deployment
    app.run(host='0.0.0.0', port=5000, debug=False)
