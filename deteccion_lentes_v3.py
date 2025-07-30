# install (one-time)
# pip install glasses-detector==1.0.3 pillow   # or newer

from glasses_detector.detector import GlassesClassifier
from PIL import Image

def glasses_detect(path):
    """
    Detects if spectacles are present in the image at the given path.
    
    Args:
        path (str): The file path to the image.
        
    Returns:
        str: 'present' if spectacles are detected, 'absent' otherwise.
    """
    # 1) Read the image
    image = Image.open(path).convert("RGB")  # ensures the image is in RGB format

    # 2) Load the pretrained binary classifier
    clf = GlassesClassifier(kind="anyglasses", size="medium")  # binary: spectacles present / absent

    # 3) Run the prediction
    prediction = clf(image)  # or simply clf(path) if you prefer

    return prediction

# # 1) Put your image’s location in a variable
# img_path = r"C:\Users\Administrador\Documents\INGENIERIA_EN_SOFTWARE\TESIS\CODIGO\funcionalidades_validador_retratos\results\image_cache\.1720858651.jpg"
# print(glasses_detector(img_path))      # → 'present' or 'absent'