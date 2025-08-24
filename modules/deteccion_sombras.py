import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from networks.sddnet import SDDNet

# Disable OpenCV multithreading (optional)
cv2.setNumThreads(0)

def get_shadow_percentage(
    image_path: str,
    ckpt_path: str = "models/sbu.ckpt",
    gpu_id: int = 0,
    im_size: int = 512
) -> float:
    """
    Loads the SDDNet model (using the given checkpoint) and returns the
    percentage of shadow pixels in the image at `image_path`.

    Args:
        image_path:  Path to the input image.
        ckpt_path:   Path to the SDDNet checkpoint (.ckpt).
        gpu_id:      CUDA device ID (falls back to CPU if unavailable).
        im_size:     Size to which the image is resized before inference.

    Returns:
        Percentage of pixels classified as shadow (0–100).
    """
    # 1) Device setup
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)

    # 2) Model initialization & checkpoint load
    model = SDDNet(
        backbone='efficientnet-b3',
        proj_planes=16,
        pred_planes=32,
        use_pretrained=True,
        fix_backbone=False,
        has_se=False,
        dropout_2d=0,
        normalize=True,
        mu_init=0.4,
        reweight_mode='manual'
    )
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.to(device).eval()

    # 3) Define transforms
    resize = transforms.Resize((im_size, im_size))
    to_tensor = transforms.ToTensor()

    # 4) Load & preprocess image
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size
    img_resized = resize(img)
    inp = to_tensor(img_resized).unsqueeze(0).to(device)

    # 5) Inference
    with torch.no_grad():
        out = model(inp)
        prob_map = torch.sigmoid(out['logit'])[0, 0].cpu()  # [H, W]

    # 6) Resize mask back to original size
    mask = transforms.Resize((orig_h, orig_w))(prob_map.unsqueeze(0))  # [1, H, W]
    mask_np = (mask.numpy()[0] * 255).astype(np.uint8)

    # 7) Binarize mask
    _, mask_bin = cv2.threshold(mask_np, 128, 255, cv2.THRESH_BINARY)

    # 8) Compute shadow percentage
    shadow_pixels = cv2.countNonZero(mask_bin)
    total_pixels  = mask_bin.shape[0] * mask_bin.shape[1]
    shadow_pct    = shadow_pixels / total_pixels * 100

    return shadow_pct

pct = get_shadow_percentage(r"C:\Users\Administrador\Documents\INGENIERIA_EN_SOFTWARE\PROYECTO_FOTOGRAFIAS_ESTUDIANTES\datasets\SBUTestNew\MyImages\ShadowHero_DSC9038.jpg")
print(f"Sombra en la imagen: {pct}%")