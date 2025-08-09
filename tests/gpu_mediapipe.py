# pip install mediapipe opencv-python
import cv2, os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# (Optional) show TFLite delegate logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

# 1) Base options: point to the .task bundle and request GPU
base_opts = python.BaseOptions(
    model_asset_path="face_landmarker.task",
    delegate=python.BaseOptions.Delegate.GPU
)

# 2) Task options
opts = vision.FaceLandmarkerOptions(
    base_options=base_opts,
    running_mode=vision.RunningMode.VIDEO,  # IMAGE | VIDEO | LIVE_STREAM
    num_faces=1,
    min_face_detection_confidence=0.5
)

# 3) Create the task
landmarker = vision.FaceLandmarker.create_from_options(opts)

# 4) Webcam loop
cap = cv2.VideoCapture(0)
ts_ms = 0
while True:
    ok, frame = cap.read()
    if not ok: break
    ts_ms += 33  # fake timestamp for VIDEO mode

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    result = landmarker.detect_for_video(mp_image, ts_ms)

    # draw simple markers
    if result.face_landmarks:
        for lm in result.face_landmarks[0]:
            x = int(lm.x * frame.shape[1]); y = int(lm.y * frame.shape[0])
            cv2.circle(frame, (x,y), 1, (0,255,0), -1)

    cv2.imshow("GPU Face Landmarker", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
