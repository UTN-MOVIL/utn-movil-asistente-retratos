import cv2
import mediapipe as mp

# Initialize MediaPipe Face Mesh and drawing utilities
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize webcam
cap = cv2.VideoCapture(0)

# Define the window name (NEW)
WINDOW_NAME = 'MediaPipe Face Mesh'

# Set up the Face Mesh model
# min_detection_confidence: Minimum confidence value for the face detection to be considered successful.
# min_tracking_confidence: Minimum confidence value for the landmarks to be considered tracked successfully.
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True, # This enables tracking of irises and lips
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        # Convert the BGR image to RGB.
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image and find face landmarks
        results = face_mesh.process(image_rgb)

        # Draw the face mesh annotations on the image.
        image.flags.writeable = True
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw the tessellation (the mesh itself)
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                
                # Draw the contours of the face
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())

                # Draw the irises (if refine_landmarks=True)
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())

        # Display the resulting frame and use the window name variable
        cv2.imshow(WINDOW_NAME, cv2.flip(image, 1))

        # --- MODIFIED SECTION ---
        # Exit loop if 'q' is pressed OR the window is closed
        key = cv2.waitKey(5) & 0xFF
        if key == ord('q') or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break
        # --- END OF MODIFIED SECTION ---

# Release the webcam and destroy all windows
cap.release()
cv2.destroyAllWindows()