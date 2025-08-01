# install (one-time)
# pip install glasses-detector==1.0.3 pillow   # or newer

from glasses_detector import GlassesClassifier
from PIL import Image

def get_glasses_probability(path, *, return_proba: bool = True, thresh: float = 0.5):
    """
    Detects spectacles in an image and, if requested, returns the probability
    that glasses are present.
    
    Args:
        path (str)           : Path to the image file.
        return_proba (bool)  : If True, return a float ∈ [0,1] with the
                               confidence; if False, return the label
                               'present' | 'absent'.  Default = True.
        thresh (float)       : Threshold used to map the probability to a
                               label when return_proba is False.
    Returns:
        float | str : Probability (0–1) or label string.
    """
    # 1) Read image
    image = Image.open(path).convert("RGB")

    # 2) Load the pretrained binary classifier
    clf = GlassesClassifier(kind="anyglasses", size="medium")

    # 3) Ask for the probability format
    proba = clf(image, format="proba")       # <-- key line 🔑
    
    if return_proba:
        return float(proba)                  # e.g. 0.87
    else:
        return "present" if proba >= thresh else "absent"

# path = r"C:\Users\Administrador\Documents\INGENIERIA_EN_SOFTWARE\TESIS\CODIGO\funcionalidades_validador_retratos\results\image_cache\0401407929.jpg"
# score = get_glasses_probability(path)                 # returns probability by default
# print(f"Probability of glasses: {score:.2%}")