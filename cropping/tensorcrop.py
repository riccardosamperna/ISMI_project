import tensorflow as tf
import matplotlib.pyplot as plt
import os
import json
import subprocess
from PIL import Image, ImageDraw
import numpy as np
from train import build_forward
from utils.train_utils import add_rectangles
import cv2
import argparse

def get_file_list(path,ext='',queue=''):
    if ext != '':
        return [os.path.join(path,f) for f in os.listdir(path) if f.endswith(''+queue+'.'+ext+'')]
    else:
        return [os.path.join(path,f) for f in os.listdir(path)]

def get_test_src(data_dir):
    test_img_dir = os.path.join(data_dir,'test')
    test_imgs_all = get_file_list(test_img_dir, 'jpg')
    return test_imgs_all

def get_train_src(data_dir, lbl_names):
    tra_imgs_all = []
    for l in lbl_names:
        tra_img_dir = os.path.join(data_dir,'train',l)
        tra_imgs_all.append(get_file_list(tra_img_dir, 'jpg'))
    return tra_imgs_all

def getBoundingBox(f):
    cor_list = []
    with open(f, 'r') as file:
        for row in file:
            name, _, x, y, w, h = row.split()
            cor_list.append((f[:17]+'/'+name[7:],[{'x1':int(x),'y1':int(y),'x2':int(x)+int(w),'y2':int(y)+int(h)}]))
    return cor_list

def drawImageRect(img_src, x1, x2, y1, y2,sizew=480,sizeh=640,box_width=2):
    print x1,x2,y1,y2
    img = Image.open(img_src)
    print sizew
    if not sizew == 0:
        img = img.resize((sizew,sizeh))
    draw = ImageDraw.Draw(img)
    for i in range(-box_width,box_width):
        draw.rectangle(((x1+i, y1+i), (x2+i, y2+i)), outline='lightgreen')
    drawImage(img)

def drawImage(img):
    fig = plt.figure(figsize=(12, 12))
    plt.imshow(img)
    plt.show()

def getSizeDistribution(l):
    d ={}
    for i, im in enumerate(l):
        with Image.open(im) as img:
            x,y =img.size
            if not img.size in d:
                d[img.size] = 1
            else:
                d[img.size] += 1
    return d

def getIndex(l,img):
    return np.where(l == img)

def getTrueBoundingBoxes(data_dir):
    # true bounding boxes
    blist1 = getBoundingBox(data_dir+"train/Type_1_bbox.tsv")
    blist2 = getBoundingBox(data_dir+"train/Type_2_bbox.tsv")
    blist3 = getBoundingBox(data_dir+"train/Type_3_bbox.tsv")
    blist = blist1 + blist2 + blist3
    print 'blist size -', len(blist)
    return np.array(blist)

def getImgWithTrueBox(true_box_list,train_flat, i):
    box_index = getIndex(true_box_list,i)
    img_index = getIndex(train_flat,i)
    cors = true_box_list[box_index[0]][0][1][0]
    img = train_flat[img_index][0]
    print cors
    print img
    return img, cors

def getAllTrainingFlat(data_dir,lbl_names):
    imgs = get_train_src(data_dir, lbl_names)
    imgs = [val for sublist in imgs for val in sublist]
    print len (imgs)
    return np.array(imgs)

def resizeImageToRatio(w):
    imgW = Image.open(w)
    width,height = imgW.size
    print width, height

    ratio = float(float(width)/float(height))
    # ratio is good
    if ratio == 0.75:
        print ('good ratio')
        return imgW
    # ratio is inverse
    if ratio == 1/0.75:
        print ('inverse ratio')
        imgW = imgW.rotate(90)
        return imgW

    # ratio is wrong
    print ratio
    ratio_new = 0.75/ratio
    print 'ratio_new',ratio_new
    if ratio < 0.75:
        print 'width'
        width *= ratio_new
    else:
        print 'height'
        height *= (1/ratio_new)
    imgc= imgW.crop((0,0,width,height))
    return imgc

class TensorCrop:
    def __init__(self,data_dir,output_dir,size_rect=(200,200)):
        print 'init'
        self.data_dir = data_dir
        self.output_dir = output_dir

        #hyper parameter file
        hypes_file = 'output/overfeat_rezoom_2017_05_23_23.01/hypes.json'
        with open(hypes_file, 'r') as f:
            self.H = json.load(f)
        # checkpoint
        self.checkpoint = 'output/overfeat_rezoom_2017_05_23_23.01/save.ckpt-3999'

        #resized dimension when trained
        self.width = self.H['image_width']
        self.height = self.H['image_height']

        # desired rect
        self.size_rect = size_rect

    def defaultBox(self):
        x = self.width/2
        y = self.height/2
        w = self.size_rect[0]/2
        h = self.size_rect[1]/2
        return x-w,x+w,y-h,y+h, self.width, self.height

    def getPos(self,rects):
        #TODO CLEAN UP

        new_w = self.width
        new_h = self.height

        if not len(rects) == 1:
            return self.defaultBox()

        x1 = rects[0].x1
        x2 = rects[0].x2
        y1 = rects[0].y1
        y2 = rects[0].y2

        if(x1<0 or x2>self.width or y1<0 or y2>self.height):
            print 'default box'
            return self.defaultBox()

        box_w = x2-x1
        box_h = y2-y1

        ratio_w = self.size_rect[0]/box_w
        ratio_h = self.size_rect[1]/box_h

        xc = (x1 + x2)/2
        yc = (y1 + y2)/2

        ratio = ratio_w if ratio_w < ratio_h else ratio_h

        # perfect content
        # new_w = self.width*ratio_w
        # new_h = self.height*ratio_h

        #isotropic rescaling
        new_w = self.width*ratio
        new_h = self.height*ratio

        #perfect content
        # xc = xc*ratio_w
        # yc = yc*ratio_h

        #isotropic rescaling
        xc = xc*ratio
        yc = yc*ratio

        x1 = xc - 100
        if x1<0:
            x1=0
        x2 = x1+200
        if x2 > new_w:
            x2 = new_w
            x1 = x2-200

        y1 = yc - 100
        if y1<0:
            y1=0
        y2 = y1+200
        if y2 > new_h:
            y2 = new_h
            y1 = y2-200

        return x1,x2,y1,y2,new_w, new_h

    def tb_predict(self,imgs):
        # rest graph
        tf.reset_default_graph()
        # input placeholder
        x_in = tf.placeholder(tf.float32, name='x_in', shape=[self.H['image_height'], self.H['image_width'], 3])

        #Tensorbox initialization
        pred_boxes, pred_logits, pred_confidences, pred_confs_deltas, pred_boxes_deltas = build_forward(self.H, tf.expand_dims(x_in, 0), 'test', reuse=None)
        grid_area = self.H['grid_height'] * self.H['grid_width']
        pred_confidences = tf.reshape(tf.nn.softmax(tf.reshape(pred_confs_deltas, [grid_area * self.H['rnn_len'], 2])), [grid_area, self.H['rnn_len'], 2])
        pred_boxes = pred_boxes + pred_boxes_deltas

        #saver
        saver = tf.train.Saver()

        # predicted images
        pred_imgs = []
        # predicted rects
        pred_rects = []
        with tf.Session() as sess:
            # init vars
            sess.run(tf.global_variables_initializer())
            #restore checkpoint
            saver.restore(sess, self.checkpoint)
            for ix, i in enumerate(imgs):
                print 'predicting box for image: ', i
                #resize for ratio
                img = resizeImageToRatio(i)
                #resize for Tensorbox
                img_r = img.resize((self.H['image_width'],self.H['image_height']))
                # convert to np.array
                img_r = np.array(img_r)
                # feed dict
                feed = {x_in: img_r}
                # run Tensorbox
                (np_pred_boxes, np_pred_confidences) = sess.run([pred_boxes, pred_confidences], feed_dict=feed)
                # stitch rect, create image with rect
                img_n, rects = add_rectangles(self.H, [img_r], np_pred_confidences, np_pred_boxes,
                                                use_stitching=True, rnn_len=self.H['rnn_len'], min_conf=0.7,
                                                show_suppressed=False)
                pred_imgs.append(img_n)
                pred_rects.append((i,img,rects))
        return pred_imgs, pred_rects

    def cropImage(self,i,img,rects):
        x1,x2,y1,y2,new_w,new_h = self.getPos(rects)
        img = img.resize((int(new_w),int(new_h)))
        img2 = img.crop((x1, y1, x2, y2))
        l = len(self.data_dir)
        img2.save(self.output_dir+i[l:])
        return img2

    def run(self,imgs):
        _,  pred = self.tb_predict(imgs)
        for i,img,rect in pred:
            self.cropImage(i,img,rect)
