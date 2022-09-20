
from PIL import Image
from glob import glob

exts = ['jpg', 'jpeg', 'png']

images = []
[images.extend(glob(f'data/minet/*/*.{ext}')) for ext in exts]

for img in images:
    image = Image.open(img)
    if image.mode != 'RGB':
        print("Before:", image.mode)
        image = image.convert('RGB')
        print("After:", image.mode)