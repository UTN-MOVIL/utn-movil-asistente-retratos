import cv2
import mediapipe as mp
import numpy as np

def detect_glasses(image_path, face_mesh_detector):
    """
    Analyzes an image to detect if a person is wearing glasses.

    Args:
        image_path (str): The file path to the input image.
        face_mesh_detector (mediapipe.solutions.face_mesh.FaceMesh):
            An initialized MediaPipe FaceMesh object.

    Returns:
        bool: True if glasses are detected, False otherwise.
              Returns False if no face is found or the image cannot be read.
    """
    # --- Image Loading ---
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read the image file: {image_path}")
            return False
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return False

    # --- Core Processing ---
    h, w, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh_detector.process(image_rgb)

    # --- Glasses Detection Logic ---
    glasses_present = False
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        
        # 1. Convert landmarks to pixel coordinates
        landmarks_px = [(int(landmark.x * w), int(landmark.y * h)) for landmark in face_landmarks.landmark]

        # 2. Define the nose bridge region of interest
        nose_bridge_idxs = [168, 6, 197, 195, 5, 4]
        nose_bridge_x = [landmarks_px[i][0] for i in nose_bridge_idxs]
        
        x_min, x_max = min(nose_bridge_x), max(nose_bridge_x)
        y_min, y_max = landmarks_px[6][1], landmarks_px[4][1]

        # 3. Crop the nose bridge area
        x_min, x_max = max(0, x_min - 5), min(w, x_max + 5)
        y_min, y_max = max(0, y_min - 5), min(h, y_max + 5)
        
        cropped_nose = image[y_min:y_max, x_min:x_max]
        
        if cropped_nose.size > 0:
            # 4. Use Canny edge detection on the cropped area
            img_blur = cv2.GaussianBlur(cropped_nose, (3, 3), sigmaX=0, sigmaY=0)
            edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)

            # 5. Check the central vertical line for edges (the glasses frame)
            center_col_idx = edges.shape[1] // 2
            if 255 in edges[:, center_col_idx]:
                glasses_present = True
                
    return glasses_present

if __name__ == "__main__":
    # Initialize MediaPipe Face Mesh once to improve performance
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )

    # --- Set Image Path ---
    # IMPORTANT: Replace this with the actual path to your image file
    image_file_path = "tests/test_image.jpg"

    # --- Run Detection and Print Result ---
    has_glasses = detect_glasses(image_path=image_file_path, face_mesh_detector=face_mesh)
    print(has_glasses)

    # --- Cleanup ---
    face_mesh.close()