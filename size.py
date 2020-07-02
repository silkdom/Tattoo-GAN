import pandas as pd
import os
import PIL
from PIL import Image
import glob

def size(img_path):
    img = Image.open(img_path)
    w,h = img.size
    if w == 1080 & h == 1080:
        return('yes')
    else:
        os.remove(img_path)
        return('no')


path = "python/sample/*.jpg"
names = []
for fname in glob.glob(path):
    if 'profile_pic' in fname:
        continue
    names.append(fname)


sizes = []
for name in range(len(names)):
    sizes.append(size(names[name]))


print('number of yes: '+str(sizes.count('yes')),'number of no: '+str(sizes.count('no')))
