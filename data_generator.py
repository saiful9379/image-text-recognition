from PIL import ImageFont, ImageDraw, Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
import codecs
from scipy import ndimage
import os

def new_data(input_text,cnt,save_dir):
    h = 64
    w = 35
    text_len =len(input_text)*12
    w = w+text_len

    image = np.zeros((h,w,3), np.uint8)
    # Convert to PIL Image
    cv2_im_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im_rgb)

    draw = ImageDraw.Draw(pil_im)

    # Choose a font
    font = ImageFont.truetype("Roboto-Regular.ttf", 25)

    # Draw the text
    draw.text((10, 10),input_text, font=font)

    # Save the image
    cv2_im_processed = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
    a = 255 - cv2_im_processed
    gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    #     gray_specl = speckle(gray)
    img_name = str(cnt)+"_"+str(input_text)+"_"+str(cnt)+".jpg"
    cv2.imwrite(save_dir+"/"+img_name,gray)


# li = []
path = "word"
if not os.path.exists(path):
    os.makedirs(path)
with open("words.txt","r") as f:
    data = f.read()
    cnt = 0
    lines = data.split("\n")
    for word in lines:
        txt = word.strip()
        refresh_txt = txt.split(" ")[-1]
        print(refresh_txt)
        new_data(refresh_txt,cnt,path)
        # li.append(txt)
        cnt +=1
        if cnt ==10000:
            break