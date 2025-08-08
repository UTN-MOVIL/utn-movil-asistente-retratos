import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

window_name = "Pose skeleton"

# Real-time webcam pose tracker
with mp_pose.Pose(
        static_image_mode=False,          # → video / webcam mode
        model_complexity=1,               # 0=Lite, 1=Full, 2=Heavy
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:

    cap = cv2.VideoCapture(0)             # default webcam
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        # Procesamiento de MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS)

        # Mostrar el frame
        cv2.imshow(window_name, frame)

        # 1️⃣ Salir con ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break

        # 2️⃣ Salir si la ventana ha sido cerrada con el botón "X"
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
