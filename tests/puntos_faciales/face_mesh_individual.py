import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True, # Process a static image
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# Load a single image file
file_name = r"C:\Users\Administrador\Documents\INGENIERIA_EN_SOFTWARE\TESIS\CODIGO\funcionalidades_validador_retratos\results\image_cache\0401775143.jpg" # <-- Put your image file name here
try:
    image = cv2.imread(file_name)
    if image is None:
        raise FileNotFoundError(f"Could not read the image file: {file_name}")
except (FileNotFoundError, Exception) as e:
    print(e)
    # Create a dummy black image if the file is not found
    image = np.zeros((512, 512, 3), np.uint8)
    cv2.putText(image, 'Image not found', (50, 250),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


# --- Core Processing ---
# Convert the BGR image to RGB before processing.
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process the image and find face landmarks
results = face_mesh.process(image_rgb)
# --- End of Core Processing ---

# The 'results.multi_face_landmarks' contains the facial points.
if results.multi_face_landmarks:
    print(f"Found {len(results.multi_face_landmarks)} face(s).")
    for face_landmarks in results.multi_face_landmarks:
        # Each 'face_landmarks' contains 478 points for a single face.
        # Example: Get the coordinates of the first landmark (point 0)
        first_landmark = face_landmarks.landmark[0]
        print(f"First landmark coordinates: (x: {first_landmark.x}, y: {first_landmark.y}, z: {first_landmark.z})")

        # You can now use these landmark coordinates for your application.
        # For visualization, you can draw them on the original image.
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        annotated_image = image.copy()
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
        )
        cv2.imshow('Face Mesh on Single Image', annotated_image)
        cv2.waitKey(0) # Wait until a key is pressed to close the window

# Clean up
face_mesh.close()
cv2.destroyAllWindows()