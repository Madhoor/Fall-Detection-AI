import os
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from matplotlib.pylab import angle
from mediapipe.tasks.python import vision
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
import math



prev_hip_y = None
fall_detected = False


MODEL_PATH = "models/pose_landmarker.task"
MODEL_URL = "https://drive.google.com/uc?export=download&id=1GsffmrIK38eHwp9u6UfstsDzlx53OUZ4"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("📥 Model not found. Downloading...")
        os.makedirs("models", exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("✅ Model downloaded successfully!")

try:
    download_model()
except Exception as e:
    print("❌ Failed to download model. Please download it manually.")
    print("Error:", e)
# Load Model        

base_options = python.BaseOptions(
    # Make sure to download the model and provide the correct path
    model_asset_path=MODEL_PATH
)

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False
)

detector = vision.PoseLandmarker.create_from_options(options)

# Video Feed

cap = cv2.VideoCapture(0) # change the argument to a video file path for processing a video instead of webcam feed / or add one for external webcam

def calculate_angle(p1, p2):
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    angle = math.degrees(math.atan2(dy, dx))
    return abs(angle)
    
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR → RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert to MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Run Pose Detection
    result = detector.detect(mp_image)
    # print(result)

    annotated_frame = frame.copy()
    if result.pose_landmarks:
        landmarks = result.pose_landmarks[0]

        # Get required landmarks
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_hip = landmarks[23]
        right_hip = landmarks[24]

        # Midpoints
        shoulder_mid_x = (left_shoulder.x + right_shoulder.x) / 2
        shoulder_mid_y = (left_shoulder.y + right_shoulder.y) / 2

        hip_mid_x = (left_hip.x + right_hip.x) / 2
        hip_mid_y = (left_hip.y + right_hip.y) / 2

        # Convert to dummy landmark-like objects
        class Point:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        shoulder_mid = Point(shoulder_mid_x, shoulder_mid_y)
        hip_mid = Point(hip_mid_x, hip_mid_y)

        # Calculate torso angle
        angle = calculate_angle(shoulder_mid, hip_mid)

        # Detect sudden drop
        if prev_hip_y is not None:
            drop = hip_mid_y - prev_hip_y
            if drop > 0.08 and angle < 45:
                fall_detected = True

        prev_hip_y = hip_mid_y

        # Draw skeleton
        drawing_utils.draw_landmarks(
            image=annotated_frame,
            landmark_list=landmarks,
            connections=vision.PoseLandmarksConnections.POSE_LANDMARKS
        )

        # Show angle on screen
        cv2.putText(annotated_frame,
                    f"Angle: {int(angle)}",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2)

        # Fall alert
        if fall_detected:
            cv2.putText(annotated_frame,
                        "FALL DETECTED",
                        (100, 200),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (0, 0, 255),
                        4)
    # print("Angle:", angle, "Fall:", fall_detected)
    # Draw landmarks if detected

    # annotated_frame = frame.copy()
    if result.pose_landmarks:
        for pose_landmarks in result.pose_landmarks:
            drawing_utils.draw_landmarks(  
                #requires the bgr image, not the rgb one
                image=annotated_frame,
                landmark_list=pose_landmarks,
                connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
                landmark_drawing_spec=drawing_styles.get_default_pose_landmarks_style()
            )
    cv2.imshow("Fall Detection Feed", annotated_frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()






