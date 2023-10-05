import PIL
from PIL import Image
import numpy as np

"""
This file takes the images on "./dataset/image_before_mask" folder. I have prepared and outlined the images to get their masks. I set the pixel treshold to 10. 
If the intensity < 10 the pixel is set to black.
"""
filename = "ring_test_2.jpg"
mask_path = f"./dataset/image_before_mask/{filename}"
mask = Image.open(mask_path)
gray = mask.convert('L')
bw = gray.point(lambda x: 0 if x < 10 else 255, '1')
bw.save(f"./dataset/test/masks/{filename}")