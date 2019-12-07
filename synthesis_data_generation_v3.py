from PIL import ImageFont, ImageDraw, Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
import codecs
from scipy import ndimage

def speckle(img):
    severity = np.random.uniform(0, 0.6)
    blur = ndimage.gaussian_filter(np.random.randn(*img.shape) * severity, 1)
    img_speck = (img + blur)
    img_speck[img_speck > 1] = 1
    img_speck[img_speck <= 0] = 0
    return img_speck


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
#     a = np.frombuffer(a, np.uint8)
# #     a.shape = (h, w, 4)
#     a = a[:, :, 0]  # grab single channel
    a = gray.astype(np.float32) / 255
    a = np.expand_dims(a, 0)
#     if rotate:
#         a = image.random_rotation(a, 3 * (w - top_left_x) / w + 1)
    a = speckle(a)
    return a,w,img_name


def get_image(txt_path,save_dir):
    output_txt = txt_path.split("/")[1]
    txt_file = save_dir.split("/")
    file_txt_save=""
    for i in txt_file[1:5]:
        file_txt_save+="/"+i

    print(file_txt_save)
    # exit()
    print(output_txt)
    with open(file_txt_save+"/"+output_txt,"w") as file_f:
        with codecs.open(txt_path, mode='r', encoding='utf-8') as f:
            lines = f.readlines()
            cnt = 0
            for line in lines:
                print(len(line),len(line.strip()))
                line = line.strip()
                h = 64
                a,w,image_name = new_data(line,cnt,save_dir)
                file_f.write(image_name+"#"+line+"\n")
                print(a.shape)
                print(image_name,"Done")
                b = a.reshape((h, w))
                cnt+=1
                # plt.imshow(b, cmap='Greys_r')
                # plt.show()

    
mono = "wordlists/wordlist_mono_clean.txt"
bi = "wordlists/wordlist_bi_clean.txt"

mono_save_directory="/media/saiful/SONGS/systhesis_data/mono_img"
bi_save_directory="/media/saiful/SONGS/systhesis_data/bi_img"

get_image(mono,mono_save_directory)
get_image(bi,bi_save_directory)