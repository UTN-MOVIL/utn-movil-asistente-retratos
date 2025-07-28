#!/usr/bin/env python3

# dlib_pytorch_inference.py
# A script to detect faces with dlib on the GPU and run inference on them with PyTorch.

import sys
from pathlib import Path

import numpy as np
import dlib
import cv2
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

# --- Configuration ---
# The script expects the model to be in a 'models' subfolder.
# Download from: http://dlib.net/files/mmod_human_face_detector.dat.bz2
try:
    BASE_DIR = Path(__file__).resolve().parent.parent
except NameError:
    BASE_DIR = Path.cwd()

MODELS_DIR = BASE_DIR / "models"
CNN_FACE_DETECTOR_PATH = MODELS_DIR / "mmod_human_face_detector.dat"
IMAGE_PATH = BASE_DIR / "test_image.jpg" # 👈 Place a test image here!

# --- Main Logic ---
def run_dlib_pytorch_inference():
    """
    Verifies dlib GPU setup and then runs face detection with dlib
    followed by inference on the detected faces using PyTorch.
    """
    print("--- Dlib GPU Detection + PyTorch Inference ---")

    # This is a critical first step.
    try:
        import dlib
    except ImportError:
        print("\n❌ CRITICAL FAILURE: The 'dlib' library is not installed.")
        sys.exit(1)

    # 1. Check if dlib was compiled with CUDA support
    print("\n[Step 1/4] Verifying dlib CUDA compilation...")
    if not dlib.DLIB_USE_CUDA:
        print("\n❌ FAILURE: Your dlib install was NOT compiled with CUDA support.")
        print("   -> Solution: Reinstall dlib from source, ensuring CMake finds your CUDA toolkit.")
        sys.exit(1)
    print("   ✅ SUCCESS: dlib was compiled with CUDA support.")

    # 2. Check for available GPU devices
    print("\n[Step 2/4] Detecting available GPU devices...")
    num_devices = dlib.cuda.get_num_devices()
    if num_devices == 0:
        print("\n❌ FAILURE: dlib did not detect any CUDA-capable GPU devices.")
        print("   -> Solution: Ensure your NVIDIA drivers are correctly installed and up to date.")
        sys.exit(1)
    print(f"   ✅ SUCCESS: dlib detected {num_devices} CUDA GPU device(s).")
    
    # Also check PyTorch GPU availability
    if not torch.cuda.is_available():
        print("\n❌ FAILURE: PyTorch did not detect any CUDA-capable GPU devices.")
        sys.exit(1)
    print(f"   ✅ SUCCESS: PyTorch detected {torch.cuda.get_device_name(0)}.")
    device = torch.device("cuda:0")


    # 3. Check if the required dlib model file exists
    print("\n[Step 3/4] Verifying dlib CNN face model...")
    if not CNN_FACE_DETECTOR_PATH.exists():
        print(f"\n❌ FAILURE: Model file not found at: {CNN_FACE_DETECTOR_PATH}")
        sys.exit(1)
    print(f"   ✅ SUCCESS: dlib model found.")

    # 4. Load models and run the full inference pipeline
    print("\n[Step 4/4] Running full Dlib + PyTorch pipeline...")
    try:
        # Load dlib's CNN face detector
        dlib_face_detector = dlib.cnn_face_detection_model_v1(str(CNN_FACE_DETECTOR_PATH))
        print("   - Dlib face detector loaded.")

        # Load a pre-trained PyTorch model (e.g., ResNet18 for demonstration)
        # In a real application, you would load your own trained model.
        pytorch_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        pytorch_model.to(device) # Move model to GPU
        pytorch_model.eval() # Set model to evaluation mode
        print("   - PyTorch ResNet18 model loaded to GPU.")

        # Define the image transformations for the PyTorch model
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Load the image with OpenCV
        if not IMAGE_PATH.exists():
            print(f"\n❌ FAILURE: Test image not found at: {IMAGE_PATH}")
            sys.exit(1)
        
        img_bgr = cv2.imread(str(IMAGE_PATH))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # dlib and PyTorch expect RGB

        # --- Dlib Part: Detect faces on GPU ---
        # The '1' indicates to upsample the image once, making it easier to find smaller faces.
        detections = dlib_face_detector(img_rgb, 1)
        print(f"\n   -> Dlib detected {len(detections)} face(s).")

        # --- PyTorch Part: Run inference on each detected face ---
        for i, d in enumerate(detections):
            face = d.rect
            print(f"   Processing Face #{i+1} at [{face.left()}, {face.top()}, {face.right()}, {face.bottom()}]")
            
            # Crop the face from the image
            cropped_face_np = img_rgb[face.top():face.bottom(), face.left():face.right()]
            
            # Convert numpy array (OpenCV format) to PIL Image for torchvision transforms
            cropped_face_pil = Image.fromarray(cropped_face_np)

            # Preprocess the image and convert to a PyTorch tensor
            input_tensor = preprocess(cropped_face_pil)
            
            # Add a batch dimension (C, H, W) -> (B, C, H, W) where B=1
            input_batch = input_tensor.unsqueeze(0)
            
            # Move the input tensor to the GPU
            input_batch = input_batch.to(device)

            # Perform inference with PyTorch
            with torch.no_grad():
                output = pytorch_model(input_batch)

            # Get the prediction
            _, pred_idx = torch.max(output, 1)
            print(f"      ✅ PyTorch Inference Result (ResNet18 class index): {pred_idx.item()}")

            # Draw rectangle on the original image for visualization
            cv2.rectangle(img_bgr, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)

        # Save or display the result
        output_path = BASE_DIR / "output_detection.jpg"
        cv2.imwrite(str(output_path), img_bgr)
        print(f"\n🎉 SUCCESS! Pipeline complete. Output image saved to: {output_path}")


    except Exception as e:
        print(f"\n❌ An error occurred during the pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_dlib_pytorch_inference()