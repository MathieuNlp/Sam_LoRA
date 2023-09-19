import PIL
from PIL import Image
import numpy as np


mask_path = "../dataset/mask_before_transform/pot3.jpg"
mask = Image.open(mask_path)
thresh = 11
fn = lambda x : 255 if x > thresh else 0
r = mask.convert('L').point(fn, mode='1')
r.save('mask.jpg')
