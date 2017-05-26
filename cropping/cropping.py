import tensorflow as tf
# import matplotlib.pyplot as plt
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
        # [,  [f for f in os.listdir(path) if f.endswith(''+queue+'.'+ext+'')]] #image name
    else:
        return [os.path.join(path,f) for f in os.listdir(path)]

def get_test_src(data_dir):
    test_img_dir = os.path.join(data_dir,'test')
    test_imgs_all = get_file_list(test_img_dir, 'jpg')
    return test_imgs_all

def drawImageRect(img_src, x1, x2, y1, y2):
    print x1,x2,y1,y2
    img = Image.open(img_src)
    img = img.resize((480,640))
    draw = ImageDraw.Draw(img)
    for i in range(-2,2):
        draw.rectangle(((x1+i, y1+i), (x2+i, y2+i)), outline='red')
    fig = plt.figure(figsize=(12, 12))
    plt.imshow(img)

class TensorCrop:
    def __init__(self,data_dir,output_dir,H,checkpoint):
        print 'init'
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.H = H
        self.checkpoint = checkpoint
        self.width = H['image_width']
        self.height = H['image_height']
        self.size_rect = (200,200)

    def defaultBox(self):
        h = self.height/2
        w = self.width/2
        return w-100,w+100,h-100,h+100, self.width, self.height

    def getPos(self,rects):
        new_w = self.width
        new_h = self.height

        if not len(rects) == 1:
            return self.defaultBox()

        x1 = rects[0].x1
        x2 = rects[0].x2
        y1 = rects[0].y1
        y2 = rects[0].y2

        if(x1<0 or x2>self.width or y1<0 or y2>self.height):
            return self.defaultBox()

        box_w = x2-x1
        box_h = y2-y1

        ratio_w = self.size_rect[0]/box_w
        ratio_h = self.size_rect[1]/box_h

        new_w = self.width*ratio_w
        new_h = self.height*ratio_h

        xc = (x1 + x2)/2
        yc = (y1 + y2)/2

        xc = xc*ratio_w
        yc = yc*ratio_h

        x1 = xc - 100
        if x1<0:
            x1=0
        x2 = x1+200
        if x2 > new_w:
            x2 = new_w
            x1 = x2-200
            # x1 -= x2-new_w??

        y1 = yc - 100
        if y1<0:
            y1=0
        y2 = y1+200
        if y2 > new_h:
            y2 = new_h
            y1 = y2-200

        return x1,x2,y1,y2,new_w, new_h
    def createnewjson(self,i,rects):

        x1,x2,y1,y2,new_w,new_h = self.getPos(rects)
        # drawImageRect(i,x1,x2,y1,y2)
        img = Image.open(i)
        img = img.resize((int(new_w),int(new_h)))
        img2 = img.crop((x1, y1, x2, y2))
        l = len(self.data_dir)
        img2.save(self.output_dir+i[l:])


    def run(self):
        print 'run'
        imgs = get_test_src(self.data_dir)
        tf.reset_default_graph()
        x_in = tf.placeholder(tf.float32, name='x_in', shape=[self.H['image_height'], self.H['image_width'], 3])

        #Tensorbox initialization
        pred_boxes, pred_logits, pred_confidences, pred_confs_deltas, pred_boxes_deltas = build_forward(H, tf.expand_dims(x_in, 0), 'test', reuse=None)
        grid_area = self.H['grid_height'] * self.H['grid_width']
        pred_confidences = tf.reshape(tf.nn.softmax(tf.reshape(pred_confs_deltas, [grid_area * H['rnn_len'], 2])), [grid_area, H['rnn_len'], 2])
        pred_boxes = pred_boxes + pred_boxes_deltas


        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, checkpoint)
            for ix, i in enumerate(imgs[:10]):
                img = Image.open(i)
                img = img.resize((self.H['image_width'],self.H['image_height']))
                img = np.array(img)
                feed = {x_in: img}
                (np_pred_boxes, np_pred_confidences) = sess.run([pred_boxes, pred_confidences], feed_dict=feed)
                new_img, rects = add_rectangles(H, [img], np_pred_confidences, np_pred_boxes,
                                                use_stitching=True, rnn_len=H['rnn_len'], min_conf=0.7,
                                                show_suppressed=False)
                self.createnewjson(i,rects)
                # fig = plt.figure(figsize=(12, 12))
                # plt.imshow(new_img)
                # plt.show()




data_dir = 'data/'
output_dir = 'testcrop/'
hypes_file = 'output/overfeat_rezoom_2017_05_23_23.01/hypes.json'
checkpoint = 'output/overfeat_rezoom_2017_05_23_23.01/save.ckpt-3999'
print 'hello'
with open(hypes_file, 'r') as f:
    H = json.load(f)

cropper = TensorCrop(data_dir,output_dir,H,checkpoint)
cropper.run()
