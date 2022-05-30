import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

def append_images(images, direction='horizontal',
                  bg_color=(255,255,255), aligment='center'):
    """
    Appends images in horizontal/vertical direction.

    Args:
        images: List of PIL images
        direction: direction of concatenation, 'horizontal' or 'vertical'
        bg_color: Background color (default: white)
        aligment: alignment mode if images need padding;
           'left', 'right', 'top', 'bottom', or 'center'

    Returns:
        Concatenated image as a new PIL image object.
    """
    widths, heights = zip(*(i.size for i in images))

    if direction=='horizontal':
        new_width = sum(widths)
        new_height = max(heights)
    else:
        new_width = max(widths)
        new_height = sum(heights)

    new_im = Image.new('RGB', (new_width, new_height), color=bg_color)


    offset = 0
    for im in images:
        if direction=='horizontal':
            y = 0
            if aligment == 'center':
                y = int((new_height - im.size[1])/2)
            elif aligment == 'bottom':
                y = new_height - im.size[1]
            new_im.paste(im, (offset, y))
            offset += im.size[0]
        else:
            x = 0
            if aligment == 'center':
                x = int((new_width - im.size[0])/2)
            elif aligment == 'right':
                x = new_width - im.size[0]
            new_im.paste(im, (x, offset))
            offset += im.size[1]

    return new_im

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

path = "/mnt/hdd1/ravikiran/idinvert_pytorch/ffhq_outputs"
images = os.listdir(path)
image_names = list(set([x.split("_")[0] for x in images if x.split("_")[-1][-3:]=="png"]))

ffhq_image_list = []
for k in image_names:
    im1 = Image.open(path+"/"+k+"_ori.png")
    im2 = Image.open(path+"/"+k+"_enc.png")
    im3 = Image.open(path+"/"+k+"_inv.png")
    out = append_images([im1,im2,im3],direction='horizontal')
    ffhq_image_list.append(out)
ffhq_concat = append_images(ffhq_image_list,direction='vertical')

path = "/mnt/hdd1/ravikiran/idinvert_pytorch/celeba_outputs"
images = os.listdir(path)
image_names = list(set([x.split("_")[0] for x in images if x.split("_")[-1][-3:]=="png"]))

celeba_image_list = []
for k in image_names:
    im1 = Image.open(path+"/"+k+"_ori.png")
    im2 = Image.open(path+"/"+k+"_enc.png")
    im3 = Image.open(path+"/"+k+"_inv.png")
    out = append_images([im1,im2,im3],direction='horizontal')
    celeba_image_list.append(out)
celeba_concat = append_images(celeba_image_list,direction='vertical')

final_img = append_images([ffhq_concat, celeba_concat], direction='horizontal')
final_img.save("/mnt/hdd1/ravikiran/ALAE/ravi_experiments/original_idinvert_ffhq_inversions/idinvert_inversion.png")

# font = cv2.FONT_HERSHEY_SIMPLEX 
  
# # org 
# org = (10, 70) 
  
# # fontScale 
# fontScale = 3
   
# # Blue color in BGR 
# color = (255, 0, 0) 
  
# # Line thickness of 2 px 
# thickness = 3

# imgs = list(os.listdir("ravi_experiments/"))
# imgs.sort()
# i = 1
# for x in batch(imgs, 5):
#     age = cv2.imread("ravi_experiments/"+x[0])
#     age = cv2.putText(age, 'Age', org, font, fontScale, color, thickness, cv2.LINE_AA)
#     attractive = cv2.imread("ravi_experiments/"+x[1])
#     attractive = cv2.putText(attractive, 'Attractive', org, font, fontScale, color, thickness, cv2.LINE_AA)
#     gender = cv2.imread("ravi_experiments/"+x[2])
#     gender = cv2.putText(gender, 'Gender', org, font, fontScale, color, thickness, cv2.LINE_AA)
#     glasses = cv2.imread("ravi_experiments/"+x[3])
#     glasses = cv2.putText(glasses, 'Glasses', org, font, fontScale, color, thickness, cv2.LINE_AA)
#     smile = cv2.imread("ravi_experiments/"+x[4])
#     smile = cv2.putText(smile, 'Smile', org, font, fontScale, color, thickness, cv2.LINE_AA)

#     age = np.concatenate((age,attractive), axis = 0)
#     age = np.concatenate((age,gender), axis = 0)
#     age = np.concatenate((age,glasses), axis = 0)
#     age = np.concatenate((age,smile), axis = 0)
#     cv2.imwrite("ravi_exps_final/{}.jpg".format(i), age)
#     i += 1

# print(imgs)