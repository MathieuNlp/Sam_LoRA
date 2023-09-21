import PIL
from PIL import Image
import numpy as np

filename = "ring4.jpg"
mask_path = f"./dataset/mask_before_transform/{filename}"
mask = Image.open(mask_path)
gray = mask.convert('L')
bw = gray.point(lambda x: 0 if x<5 else 255, '1')
bw.save(f"./dataset/train/masks/{filename}")