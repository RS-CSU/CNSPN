from PIL import Image
import os



from glob import glob
dir = '/media/gy/study/paper_experment/01paper/datasets/RSD46-WHU/images_background/*/*.png'
paths = glob(dir)
i = 0
for path in paths:
    i = i+1
    print(i, path)
    Image.open(path).convert('RGB')