from ultralytics import YOLO
import cv2

# Load trained classification model
model = YOLO("runs/classify/train/weights/best.pt")

# Disease → solution mapping
solutions = {
    "Potato___Early_blight": "Spray Mancozeb or Chlorothalonil every 7 days",
    "Potato___Late_blight": "Use Copper fungicide and remove infected leaves",
    "Potato___Healthy": "Plant is healthy. No action needed"
}

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    probs = results[0].probs

    cls_id = probs.top1
    confidence = probs.top1conf
    disease = results[0].names[cls_id]

    solution = solutions.get(disease, "Consult agriculture expert")

    cv2.putText(frame, f"Disease: {disease}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(frame, f"Confidence: {confidence:.2f}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.putText(frame, f"Solution: {solution}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow("Plant Disease Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
