from PIL import Image
import glob
image_list = []

import os

#to get the current working directory
directory = os.getcwd()

print(directory)
for filename in glob.glob('/home/iyer/Detectron2/simple_images/aerail_images_of_traffic/*.jpg'): #assuming gif
    print("Reading:", filename)
    im=Image.open(filename)
    rgb_im = im.convert('RGB')
    rgb_im.save(filename)    
    image_list.append(im)
    
print("Conversion completed:", len(image_list))

