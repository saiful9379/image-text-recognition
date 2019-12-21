######################################################################################
# Discription: Image sequence recognization use CNN+LSTM and CTC loss
# Author : Saiful islam
# Email : saifulbrur79@gmail.com
# Copyright : saiful79.github.io
# license: MIT
######################################################################################
import os
import itertools
import codecs
import re
import datetime
import cairocffi as cairo
import editdistance
import numpy as np
from scipy import ndimage
import pylab
import string
from keras import backend as K
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation
from keras.layers import Reshape, Lambda
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.layers.recurrent import GRU
from keras.optimizers import SGD,Adam
from keras.utils.data_utils import get_file
from keras.preprocessing import image
import keras.callbacks
from keras.models import load_model
# from keras.preprocessing.image import load_img,img_to_array
import matplotlib.pyplot as plt
import cv2
import glob

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

# # Reverse translation of numerical classes back to characters
# def labels_to_text(labels):
#     ret = []
#     for c in labels:
#         if c == len(alphabet):  # CTC Blank
#             ret.append("")
#         else:
#             ret.append(alphabet[c])
#     return "".join(ret)
# Reverse translation of numerical classes back to characters
def labels_to_text(labels):
    ret = []
    for c in labels:
        if c == len(alphabet):  # CTC Blank
            ret.append("")
        else:
            ret.append(alphabet[c])
    return "".join(ret)

alphabet = string.digits + string.punctuation +'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ '
# alphabet = alphabet+" "

absolute_max_string_len=20

# character_count = 28

def get_output_size():
    return len(alphabet)+1

weight_file = 'weights-fine-tune232-loss-0.000284.h5'
img_w = 128
# Input Parameters
img_h = 64
# words_per_epoch = 16000
# val_split = 0.2
# val_words = int(words_per_epoch * (val_split))

# Network parameters
conv_filters = 16
kernel_size = (3, 3)
pool_size = 2
time_dense_size = 32
rnn_size = 512
minibatch_size = 32

if K.image_data_format() == 'channels_first':
    input_shape = (1, img_w, img_h)
else:
    input_shape = (img_w, img_h, 1)


act = 'relu'

input_data = Input(name='the_input', shape=input_shape, dtype='float32')
inner = Conv2D(conv_filters, kernel_size, padding='same',
                activation=act, kernel_initializer='he_normal',
                name='conv1')(input_data)
inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max1')(inner)
inner = Conv2D(conv_filters, kernel_size, padding='same',
                activation=act, kernel_initializer='he_normal',
                name='conv2')(inner)
inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max2')(inner)
# inner = Dropout(0.3)(inner)
# conv3 added
inner = Conv2D(16, (2, 2), padding='same',
                activation=act, kernel_initializer='he_normal',
                name='conv3')(inner)
# inner = BatchNormalization()(inner)
# inner = Dropout(0.2)(inner)

conv_to_rnn_dims = (img_w // (pool_size ** 2),
                    (img_h // (pool_size ** 2)) * conv_filters)
inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)

# cuts down input size going into RNN:
inner = Dense(time_dense_size, activation=act, name='dense1')(inner)

# Two layers of bidirectional GRUs
# GRU seems to work as well, if not better than LSTM:
gru_1 = GRU(rnn_size, return_sequences=True,
            kernel_initializer='he_normal', name='gru1')(inner)
gru_1b = GRU(rnn_size, return_sequences=True,
              go_backwards=True, kernel_initializer='he_normal',
              name='gru1_b')(inner)
gru1_merged = add([gru_1, gru_1b])
gru_2 = GRU(rnn_size, return_sequences=True,
            kernel_initializer='he_normal', name='gru2')(gru1_merged)
gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True,
              kernel_initializer='he_normal', name='gru2_b')(gru1_merged)

# transforms RNN output to character activations:
inner = Dense(get_output_size(), kernel_initializer='he_normal',name='dense2')(concatenate([gru_2, gru_2b]))
y_pred = Activation('softmax', name='softmax')(inner)
# Model(inputs=input_data, outputs=y_pred).summary()
# exit()

labels = Input(name='the_labels',shape=[absolute_max_string_len], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')
# Keras doesn't currently support loss funcs with extra parameters
# so CTC loss is implemented in a lambda layer
loss_out = Lambda(ctc_lambda_func, output_shape=(1,),name='ctc')([y_pred, labels, input_length, label_length])
lr = 0.001 / 30.0 # decrease 20 % 
optimizer = Adam(lr=lr)
model = Model(inputs=[input_data, labels, input_length, label_length],outputs=loss_out)
# the loss calc occurs elsewhere, so use a dummy lambda func for the loss
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer)
model.load_weights(weight_file)

# captures output of softmax so we can decode the output during visualization
test_func = K.function([input_data], [y_pred])

model_p = Model(inputs=input_data, outputs=y_pred)

def decode_predict_ctc(out, top_paths = 1):
  results = []
  beam_width = 5
  if beam_width < top_paths:
    beam_width = top_paths
  for i in range(top_paths):
    lables = K.get_value(K.ctc_decode(out, input_length=np.ones(out.shape[0])*out.shape[1],greedy=False, beam_width=beam_width, top_paths=top_paths)[0][i])[0]
    # print("label:",lables)
    # exit()
    text = labels_to_text(lables)
    results.append(text)
  return results
  
def predit_a_image(a, top_paths = 1):
  c = np.expand_dims(a.T, axis=0)
  net_out_value = model_p.predict(c)
  top_pred_texts = decode_predict_ctc(net_out_value, top_paths)
  return top_pred_texts

def paint_text(image_file_name, w, h, rotate=False, ud=False, multi_fonts=False):
    img = cv2.imread(image_file_name)
    gray = img 
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray,(w,h))
    gray = gray[:, :, 0]
    a = gray.astype(np.float32) / 255
    a = np.expand_dims(a, axis=0)
    return a


if __name__ =="__main__":
  h = 64
  w = 128
  # img_path = "5_forks_30346.jpg"
  img_path = glob.glob("test/*.jpg")
  for img in img_path:
    a = paint_text(img,w,h)
    c = np.expand_dims(a.T, axis=0)
    net_out_value = model_p.predict(c)
    pred_texts = decode_predict_ctc(net_out_value)
    print(pred_texts,img.split("/")[-1])
