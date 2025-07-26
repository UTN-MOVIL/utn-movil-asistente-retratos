import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import time
from insightface.app import FaceAnalysis
import onnxruntime as ort

class FaceRecognitionApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Face Recognition System")
        self.window.geometry("800x500")
        
        # Set up ONNX Runtime options
        ort.set_default_logger_severity(3)  # Reduce logging noise
        
        # Check available providers and select appropriate one
        available_providers = ort.get_available_providers()
        
        # Default to DirectML if no NVIDIA GPU is found
        if 'CUDAExecutionProvider' in available_providers:
            self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.status_message = "Using NVIDIA GPU with CUDA"
        else:
            self.providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
            self.status_message = "Using DirectML (no NVIDIA GPU found)"
            
        # Initialize FaceAnalysis with selected providers
        self.app = FaceAnalysis(name="buffalo_s", providers=self.providers)
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
        # Initialize variables
        self.camera = None
        self.is_running = False
        self.target_embedding = None
        self.target_image_path = ""
        self.similarity_threshold = 0.53  # Threshold for face match
        
        # Configure the main window layout
        self.window.columnconfigure(0, weight=1)
        self.window.columnconfigure(1, weight=1)
        self.window.rowconfigure(0, weight=1)
        self.window.rowconfigure(1, weight=0)
        
        # Create a frame for the camera feed on the left
        self.camera_frame = ttk.Frame(self.window)
        self.camera_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # Create a canvas for displaying the camera feed
        self.camera_canvas = tk.Canvas(self.camera_frame, bg="black")
        self.camera_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Create a frame for controls on the right
        self.control_frame = ttk.Frame(self.window)
        self.control_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        
        # Center the controls in the control frame
        self.control_frame.columnconfigure(0, weight=1)
        for i in range(5):  # Create rows for controls
            self.control_frame.rowconfigure(i, weight=1)
        
        # Create the "Select Target Image" button
        self.select_button = ttk.Button(
            self.control_frame, 
            text="Select Target Image",
            command=self.select_target_image
        )
        self.select_button.grid(row=0, column=0, pady=10)
        
        # Create status label
        self.status_label = ttk.Label(self.control_frame, text="No target image selected")
        self.status_label.grid(row=1, column=0, pady=5)
        
        # Create target image preview (will be updated when a target is selected)
        self.target_preview_label = ttk.Label(self.control_frame, text="Target Image Preview")
        self.target_preview_label.grid(row=2, column=0, pady=5)
        self.target_preview = ttk.Label(self.control_frame)
        self.target_preview.grid(row=3, column=0, pady=5)
        
        # Create a status bar at the bottom
        self.status_bar = ttk.Label(self.window, text=f"Ready - {self.status_message}", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.grid(row=1, column=0, columnspan=2, sticky="ew")
        
        # Start the camera
        self.start_camera()
    
    def start_camera(self):
        """Initialize and start the camera feed"""
        self.camera = cv2.VideoCapture(0)  # 0 is usually the default camera
        if not self.camera.isOpened():
            self.show_error_message("Error: Could not open camera")
            return
        
        self.is_running = True
        # Start camera feed in a separate thread
        self.camera_thread = threading.Thread(target=self.update_camera)
        self.camera_thread.daemon = True
        self.camera_thread.start()
    
    def update_camera(self):
        """Update the camera feed continuously with face detection"""
        while self.is_running:
            ret, frame = self.camera.read()
            if ret:
                # Process frame for face detection
                display_frame = frame.copy()
                
                # If we have a target embedding, detect and compare faces
                if self.target_embedding is not None:
                    face_results = self.get_face_embeddings(frame)
                    for face_img, embedding, (x1, y1, x2, y2) in face_results:
                        similarity = self.compare_face_embeddings(embedding, self.target_embedding)
                        label = f"Sim: {similarity:.2f}"
                        color = (0, 255, 0) if similarity > self.similarity_threshold else (0, 0, 255)
                        
                        # Draw rectangle and similarity score
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Add match indicator if above threshold
                        if similarity > self.similarity_threshold:
                            label += " ✔ Match"
                        
                        # Display text with a darker background for readability
                        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        cv2.rectangle(display_frame, (x1, y1-text_size[1]-10), (x1+text_size[0], y1), (0, 0, 0), -1)
                        cv2.putText(display_frame, label, (x1, y1-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Convert the frame from BGR (OpenCV format) to RGB
                frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                
                # Convert to PhotoImage format for Tkinter
                image = Image.fromarray(frame_rgb)
                
                # Resize to fit the canvas
                canvas_width = self.camera_canvas.winfo_width()
                canvas_height = self.camera_canvas.winfo_height()
                
                if canvas_width > 1 and canvas_height > 1:  # Ensure valid dimensions
                    image = image.resize((canvas_width, canvas_height), Image.LANCZOS)
                
                photo = ImageTk.PhotoImage(image=image)
                
                # Update the canvas with the new image
                self.camera_canvas.create_image(0, 0, image=photo, anchor=tk.NW)
                self.camera_canvas.image = photo  # Keep a reference to prevent garbage collection
            
            time.sleep(0.03)  # Approximately 30 FPS
    
    def get_face_embeddings(self, frame):
        """
        Get face embeddings from a frame using InsightFace.
        
        Returns:
            list: List of tuples (face_image, embedding, bbox) for each detected face
        """
        # Process in batches for better GPU utilization
        faces = self.app.get(frame, max_num=0)  # 0 means detect all faces
        
        # Process and return detected faces with embeddings
        results = []
        for face in faces:
            # Get face coordinates
            x1, y1, x2, y2 = map(int, face.bbox)
            
            if x2 - x1 > 0 and y2 - y1 > 0:  # Ensure valid dimensions
                # Extract face region
                face_img = frame[y1:y2, x1:x2].copy()
                
                # Get embedding
                embedding = face.embedding
                
                # Store result
                results.append((face_img, embedding, (x1, y1, x2, y2)))
        
        return results
    
    def compare_face_embeddings(self, embedding1, embedding2):
        """
        Compare two face embeddings using cosine similarity.
        
        Returns:
            float: Cosine similarity score between 0 and 1.
                   Higher values indicate greater similarity (1 = identical).
        """
        # Check if embeddings are not empty
        if embedding1 is None or embedding2 is None:
            return 0.0
        
        # Ensure embeddings are numpy arrays
        if not isinstance(embedding1, np.ndarray):
            embedding1 = np.array(embedding1)
        if not isinstance(embedding2, np.ndarray):
            embedding2 = np.array(embedding2)
            
        # Calculate norm products
        norm_product = np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        
        # Prevent division by zero
        if norm_product == 0:
            return 0.0
            
        # Calculate cosine similarity
        cos_sim = np.dot(embedding1, embedding2) / norm_product
        
        # Cosine similarity normally ranges from -1 to 1
        # For face recognition, we often want a score between 0 and 1
        # where 1 is identical match
        return float(max(0, cos_sim))
    
    def select_target_image(self):
        """Handle the select target image button press"""
        # Open file dialog
        file_path = filedialog.askopenfilename(
            title="Select Target Face Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.target_image_path = file_path
            # Load and process the target image
            target_image = cv2.imread(self.target_image_path)
            target_faces = self.get_face_embeddings(target_image)
            
            if target_faces:
                # Get the first detected face
                face_img, embedding, _ = target_faces[0]
                self.target_embedding = embedding
                self.status_label.config(text="Target face loaded successfully")
                
                # Update preview image
                preview_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                preview_img = Image.fromarray(preview_img)
                
                # Resize for display
                preview_img = preview_img.resize((150, 150), Image.LANCZOS)
                photo = ImageTk.PhotoImage(preview_img)
                
                self.target_preview.config(image=photo)
                self.target_preview.image = photo  # Keep a reference
            else:
                self.status_label.config(text="No face detected in target image")
                self.target_embedding = None
    
    def show_error_message(self, message):
        """Display an error message"""
        messagebox.showerror("Error", message)
    
    def on_close(self):
        """Clean up resources when closing the application"""
        self.is_running = False
        if self.camera is not None:
            self.camera.release()
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()