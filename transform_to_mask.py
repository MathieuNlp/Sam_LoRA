import PIL
from PIL import Image
import numpy as np


mask_path = "../dataset/mask_before_transform/jam5.jpg"
mask = Image.open(mask_path)
thresh = 10
fn = lambda x : 255 if x > thresh else 0
r = mask.convert('L').point(fn, mode='1')
r.save('mask.jpg')
