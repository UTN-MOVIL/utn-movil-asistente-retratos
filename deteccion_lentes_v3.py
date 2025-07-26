# install (one-time)
# pip install glasses-detector==1.0.3 pillow   # or newer

from glasses_detector import GlassesClassifier
from PIL import Image

# 1) Put your image’s location in a variable
img_path = r"C:\Users\Administrador\Documents\INGENIERIA_EN_SOFTWARE\PROYECTO_FOTOGRAFIAS_ESTUDIANTES\datasets\validated_color\0104651666.jpg"

# 2) Read the image
image = Image.open(img_path).convert("RGB")     # makes sure it’s RGB

# 3) Load the pretrained binary classifier
clf = GlassesClassifier(kind="anyglasses", size="medium")  # binary: spectacles present / absent

# 4) Run the prediction
prediction = clf(image)        # or simply clf(img_path) if you prefer

print(f"Spectacles detected: {prediction}")      # → 'present' or 'absent'
