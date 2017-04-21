from __future__ import division
import h5py
import sys
import numpy as np
import matplotlib.pylab as plt
from  os import listdir
import pickle
import os
import glob
import math
import time
import cv2
from skimage import io

from imgloader import load_single_img
from keras.preprocessing.image import img_to_array
import json

def load_train(use_cached=True,filepath='train.hdf5',data_dir='./input/train',crop_rows=400,crop_cols=400,no=1480,mode="resize"):
    cervixTypes = ['Type_1','Type_2','Type_3']   

    num_total_images = no
    if use_cached is False:

        print('create new hdf5 file')
        file = h5py.File(filepath, "w")
        images = file.create_dataset("images", (num_total_images, 3, crop_rows, crop_cols), chunks=True, dtype='f', compression="lzf")
        targets = file.create_dataset("targets", (num_total_images, 3), chunks=True, dtype='int32')
        
        print('Read train images')
        total = 0
        for i,d in enumerate(cervixTypes): #parse all subdirections
            sys.stdout.write(".")
            sys.stdout.flush()

            files = listdir(os.path.join(data_dir, d))  
            for j, f in enumerate(files):           #parse through all files
                print("Cervix #", i+1, ": ", cervixTypes[i], ", image # ", j+1, ": ", f)
                if not(f == '.DS_Store'):
                    current_img = load_single_img(data_dir+"/"+d+"/"+f)

                    if mode is "resize":
                        current_img = current_img.astype('float32')
                        current_img /= 255
                        current_img = cv2.resize(current_img, (crop_cols, crop_rows))
                    print(current_img.shape)
                    current_img = current_img.transpose((2, 0, 1))
                    print(current_img.shape)

                    images[total, :, :, :] = current_img
                    targets[total, :] = 0
                    targets[total, i] = 1
                    total += 1
            file.flush()
    else:
        print('load from hdf5 file')
        file = h5py.File(filepath, "r")

        images = file["images"]
        targets = file["targets"]

    sys.stdout.write('\n Doooone :)\n')
    return images, targets

if __name__=='__main__':
  load_train(filepath='train_channel.hdf5', use_cached=False)
  #load_test(filepath='test_big.hdf5', data_dir='data/test_stg2', use_cached=False, mode='resize')