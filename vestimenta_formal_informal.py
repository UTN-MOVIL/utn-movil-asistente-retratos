# verificar que tal esta este modelo: https://universe.roboflow.com/damit/dress-code-4tlyd

from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

# Cargar el modelo de fashion-clip
model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")

# Lista de opciones (etiquetas en lenguaje natural)
text_inputs = ["ropa formal", "ropa informal"]

# Cargar la imagen que quieres analizar
image = Image.open(r"C:\Users\Administrador\Downloads\look-rapera-600x900.jpg").convert("RGB")  # asegúrate que sea RGB

# Procesar la imagen y los textos
inputs = processor(text=text_inputs, images=image, return_tensors="pt", padding=True)

# Obtener predicciones
outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # similitudes
probs = logits_per_image.softmax(dim=1)      # convertir a probabilidades

# Mostrar resultado
for label, prob in zip(text_inputs, probs[0]):
    print(f"{label}: {prob:.2%}")

# Resultado final
predicted_label = text_inputs[probs.argmax()]
print(f"\n➡️ La prenda es: {predicted_label}")
