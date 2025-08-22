import cv2
import torch
import time

# Load YOLOv5s model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Open camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Camera not opening. Check permissions and try again.")
    exit()

# Timer setup
start_time = None
time_limit = 15  # seconds
force_green = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    # Run detection
    results = model(frame)

    # Get only person detections (class 0)
    people = [d for d in results.pred[0] if int(d[-1]) == 0]
    num_people = len(people)

    # Timer control
    if num_people >= 1 and start_time is None:
        start_time = time.time()

    elapsed = int(time.time() - start_time) if start_time else 0

    # Force green if time limit passed and at least 1 person
    if start_time and elapsed >= time_limit and num_people < 3:
        force_green = True

    # Determine color logic
    if num_people == 0:
        color = (0, 0, 255)  # Red
        force_green = False
        start_time = None  # Reset timer if no people
    elif num_people >= 3 or force_green:
        color = (0, 255, 0)  # Green
    else:
        color = (0, 0, 255)  # Red

    # Draw info on frame
    label = f"People Detected: {num_people}"
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Draw circle signal
    cv2.circle(frame, (50, 70), 20, color, -1)

    # Show timer next to the circle
    timer_text = f"{elapsed}s" if start_time else "0s"
    cv2.putText(frame, timer_text, (80, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Display frame
    cv2.imshow("Lift Detector Demo", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("✅ Quit requested.")
        break

cap.release()
cv2.destroyAllWindows()
