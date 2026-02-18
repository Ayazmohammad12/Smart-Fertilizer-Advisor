import cv2
import numpy as np
import time
from collections import deque
from flask import Flask, Response, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

DISPLAY_DURATION = 10  # seconds hold after leaf removed
VOTE_FRAMES = 7        # smoothing window

DISEASE_DB = {
    "HEALTHY": {
        "title": "STATUS: HEALTHY PLANT",
        "fert": ["Nitrogen: NORMAL","Phosphorus: NORMAL","Potassium: NORMAL","Apply: NPK 19-19-19"],
        "tip": "Plant is healthy. Maintain nutrients.",
        "color": (0,255,0)
    },
    "N_DEF": {
        "title": "STATUS: NITROGEN DEFICIENCY",
        "fert": ["Nitrogen: LOW","Apply: Urea","Add compost"],
        "tip": "Yellowing leaf due to Nitrogen deficiency.",
        "color": (0,255,255)
    },
    "FE_DEF": {
        "title": "STATUS: IRON DEFICIENCY",
        "fert": ["Iron: LOW","Apply: Ferrous Sulphate"],
        "tip": "Yellow with green veins indicates Iron deficiency.",
        "color": (0,200,255)
    },
    "K_DEF": {
        "title": "STATUS: POTASSIUM DEFICIENCY",
        "fert": ["Potassium: LOW","Apply: Potash"],
        "tip": "Brown edges indicate Potassium deficiency.",
        "color": (0,140,255)
    },
    "P_DEF": {
        "title": "STATUS: PHOSPHORUS DEFICIENCY",
        "fert": ["Phosphorus: LOW","Apply: DAP"],
        "tip": "Purple patches indicate Phosphorus deficiency.",
        "color": (200,0,200)
    },
    "MG_DEF": {
        "title": "STATUS: MAGNESIUM DEFICIENCY",
        "fert": ["Magnesium: LOW","Apply: Epsom salt"],
        "tip": "Yellow mottling indicates Magnesium deficiency.",
        "color": (180,180,0)
    },
    "ZN_DEF": {
        "title": "STATUS: ZINC DEFICIENCY",
        "fert": ["Zinc: LOW","Apply: Zinc Sulphate"],
        "tip": "White patches indicate Zinc deficiency.",
        "color": (255,255,200)
    },
    "FUNGAL": {
        "title": "STATUS: FUNGAL / RUST DISEASE",
        "fert": ["Spray: Mancozeb","Reduce humidity"],
        "tip": "Dark or rust spots indicate fungal disease.",
        "color": (0,0,255)
    }
}

current_state = "IDLE"
active_info = None
current_recommendation = {"plant":"Scanning...","disease":"None","solution":"Align Leaf"}
leaf_present = False
last_leaf_time = 0

status_history = deque(maxlen=VOTE_FRAMES)

def draw_ui(frame, box, state, info, timer_val, debug_text=""):
    x1,y1,x2,y2 = box
    h,w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay,(w-400,0),(w,h),(20,20,20),-1)
    cv2.addWeighted(overlay,0.7,frame,0.3,0,frame)

    color=(0,255,255)
    if state=="LOCKED" and info: color=info['color']
    if state=="IDLE": color=(0,0,255)

    l=40
    cv2.line(frame,(x1,y1),(x1+l,y1),color,3)
    cv2.line(frame,(x1,y1),(x1,y1+l),color,3)
    cv2.line(frame,(x2,y2),(x2-l,y2),color,3)
    cv2.line(frame,(x2,y2),(x2,y2-l),color,3)

    bx=w-370
    cv2.putText(frame,"PLANT SCANNER",(bx,50),1,1.5,(200,200,200),2)
    cv2.putText(frame,f"DEBUG: {debug_text}",(20,h-20),1,0.8,(255,255,255),1)

    if state=="LOCKED" and info:
        cv2.putText(frame,info['title'],(bx,120),1,1.2,color,2)
        cv2.putText(frame,"ACTION:",(bx,180),1,1.1,(255,255,255),1)
        y=220
        for f in info['fert']:
            cv2.putText(frame,"> "+f,(bx,y),1,1.0,(200,200,200),1)
            y+=35
        cv2.putText(frame,info['tip'],(bx,y+20),1,0.9,color,1)
    elif state=="SCANNING":
        cv2.putText(frame,"ANALYZING...",(bx,120),1,2.0,(0,255,255),2)
    else:
        cv2.putText(frame,"PLACE A LEAF",(bx,120),1,1.5,(0,0,255),2)

    return frame

def classify_leaf(hsv):
    green = cv2.inRange(hsv,(25,40,40),(90,255,255))
    yellow = cv2.inRange(hsv,(15,50,50),(35,255,255))
    brown = cv2.inRange(hsv,(5,50,20),(15,255,200))
    white = cv2.inRange(hsv,(0,0,200),(180,40,255))
    dark = cv2.inRange(hsv,(0,0,0),(180,255,50))
    purple = cv2.inRange(hsv,(125,50,50),(160,255,255))

    total = hsv.shape[0]*hsv.shape[1]

    gp = cv2.countNonZero(green)/total
    yp = cv2.countNonZero(yellow)/total
    bp = cv2.countNonZero(brown)/total
    wp = cv2.countNonZero(white)/total
    dp = cv2.countNonZero(dark)/total
    pp = cv2.countNonZero(purple)/total

    if gp > 0.45:
        return "HEALTHY","Mostly Green"
    if pp > 0.10:
        return "P_DEF","Purple detected"
    if yp > 0.35:
        return "N_DEF","Yellow dominant"
    if yp > 0.20 and gp > 0.15:
        return "FE_DEF","Yellow + green veins"
    if bp > 0.20:
        return "K_DEF","Brown edges"
    if wp > 0.20:
        return "ZN_DEF","White patches"
    if dp > 0.15:
        return "FUNGAL","Dark spots"
    return "MG_DEF","Mixed colors"

def generate_frames():
    global current_state, active_info, current_recommendation
    global leaf_present, last_leaf_time

    cap=cv2.VideoCapture(0)
    cap.set(3,1280); cap.set(4,720)

    while True:
        ret,frame = cap.read()
        if not ret: break
        frame=cv2.flip(frame,1)

        box_s=400
        x1,y1=100,(frame.shape[0]-box_s)//2
        x2,y2=x1+box_s,y1+box_s
        roi=frame[y1:y2,x1:x2]

        hsv=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv,(25,40,40),(90,255,255))
        contours,_ = cv2.findContours(green_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        if len(contours)>0 and cv2.contourArea(max(contours,key=cv2.contourArea))>5000:
            leaf_present=True
            last_leaf_time=time.time()

            status,debug = classify_leaf(hsv)
            status_history.append(status)

            final_status = max(set(status_history), key=status_history.count)

            active_info = DISEASE_DB[final_status]
            current_state="LOCKED"
            current_recommendation={"plant":"LEAF","disease":active_info['title'],"solution":active_info['tip']}

        else:
            if time.time()-last_leaf_time < DISPLAY_DURATION and active_info:
                current_state="LOCKED"
                debug="Holding last result"
            else:
                current_state="IDLE"
                active_info=None
                current_recommendation={"plant":"Scanning...","disease":"None","solution":"Align Leaf"}
                status_history.clear()
                debug="No leaf"

        frame=draw_ui(frame,(x1,y1,x2,y2),current_state,active_info,0,debug)
        _,buffer=cv2.imencode('.jpg',frame)
        yield(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'+buffer.tobytes()+b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return """
    <body style="background:#000; color:white; font-family:sans-serif; text-align:center;">
    <h1 style="color:#0f0;">🌿 SIMPLE PLANT MONITOR 🌿</h1>
    <div style="display:flex; justify-content:center;">
    <img src="/video_feed" style="width:70%; border:4px solid #333; border-radius:15px;">
    </div>
    <div id="data" style="margin-top:20px; font-size:24px;"></div>
    <script>
    setInterval(()=>{fetch('/detection_data').then(r=>r.json()).then(d=>{
    let color=d.disease.includes("HEALTHY")?"#0f0":"#f00";
    document.getElementById('data').innerHTML=
    `<span style="color:${color}; font-weight:bold;">${d.disease}</span><br>
    <span style="font-size:18px; color:#ccc;">${d.solution}</span>`;
    });},500);
    </script></body>
    """

@app.route('/detection_data')
def detection_data():
    return jsonify(current_recommendation)

if __name__=="__main__":
    app.run(host='0.0.0.0',port=5000,threaded=True)
