from load_data import load_train_data, load_train_data2, load_test
from utils import *
# import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
from PIL import Image
import datetime
import os
import csv



class simpleModel:

    def __init__(self,
                img_size=200,
                num_classes=3,
                num_channels=3,
                batch_size=64):
        print num_classes

        self.img_size = img_size
        self.img_shape = (self.img_size, self.img_size)
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.batch_size = batch_size

        self.sess = tf.Session()
        self.x = tf.placeholder(tf.float32, shape=[None, self.img_size, self.img_size,self.num_channels], name='x')
        self.y_true = tf.placeholder(tf.float32, shape=[None, self.num_classes], name='y_true')
        self.y_true_cls = tf.argmax(self.y_true, dimension=1)

        self.total_iterations = 0
        self.init_model()

        # saver
        self.saver = tf.train.Saver()


    def init_model(self):
        # conv 1
        W_conv1 = weight_variable([3, 3, 3, 16])
        b_conv1 = bias_variable([16])
        net = tf.nn.relu(conv2d(self.x, W_conv1) + b_conv1)
        net = max_pool_2x2(net)
        # conv 2
        W_conv2 = weight_variable([3, 3, 16, 32])
        b_conv2 = bias_variable([32])
        net = tf.nn.relu(conv2d(net, W_conv2) + b_conv2)
        net = max_pool_2x2(net)
        # conv 3
        W_conv3 = weight_variable([3, 3, 32, 64])
        b_conv3 = bias_variable([64])
        net = tf.nn.relu(conv2d(net, W_conv3) + b_conv3)
        net = max_pool_2x2(net)
        # conv 4
        W_conv4 = weight_variable([3, 3, 64, 128])
        b_conv4 = bias_variable([128])
        net = tf.nn.relu(conv2d(net, W_conv4) + b_conv4)
        net = max_pool_2x2(net)
        # reshape
        shape = int(np.prod(net.get_shape()[1:]))
        net = tf.reshape(net, shape=[-1, shape])  # flatten
        # fc1
        W_fc1 = weight_variable([shape, 128])
        b_fc1 = bias_variable([128])
        net = tf.nn.relu(tf.matmul(net, W_fc1) + b_fc1)
        # fc2
        W_fc2 = weight_variable([128, self.num_classes])
        b_fc2 = bias_variable([self.num_classes])
        net = tf.matmul(net, W_fc2) + b_fc2
        self.net = net

        #softmax predicted
        self.y_pred = tf.nn.softmax(net)
        #class predicted output
        self.y_pred_cls = tf.argmax(self.y_pred, dimension=1)

        #softmax loss array
        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.net,labels=self.y_true)
        #loss/cost
        self.cost = tf.reduce_mean(self.cross_entropy)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.cost)

        self.correct_prediction = tf.equal(self.y_pred_cls, self.y_true_cls)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        self.sess.run(tf.global_variables_initializer())


    def optimize(self,num_iterations,classes=3,save=False):
        train_ext, _ = load_train_data(self.batch_size,self.num_classes)

        checkpoint="ckpt"
        logfile="train_log"
        # log/training report
        log = {}
        # log['start_time'] = time.ctime()
        # log['alpha'] = alpha
        # log['batch_size'] = batch_size
        # log['steps'] = steps
        log['checkpoint'] = checkpoint
        log['loss'] = []

        ckpt_dir = 'checkpoints/'+str(datetime.datetime.now())+'/'

        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

        start_time = time.time()
        for i in range(self.total_iterations,self.total_iterations + num_iterations):
            print "training:", i, "/", self.total_iterations
            #get batch
            x_batch, y_true_batch = train_ext.get_random_batch_balanced()
            #set feed_dict
            feed_dict_train = {self.x: x_batch,self.y_true: y_true_batch}
            #run session
            _,acc,loss = self.sess.run([self.optimizer,self.accuracy,self.cost], feed_dict=feed_dict_train)

            print 'acc: ', acc*100, 'loss: ',loss
            if save and i % 500 == 0:
                self.save(ckpt_dir+'c-'+str(classes)+'-itt-'+str(self.total_iterations))

        self.total_iterations += num_iterations
        end_time = time.time()
        time_dif = end_time - start_time
        print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
        # save final model
        if save:
            self.save(ckpt_dir+'c-'+str(classes)+'-itt-'+str(self.total_iterations))
            # save log
            #np.save(logfile, log)

    def validation(self,num_iterations):
        _, val_ext = load_train_data(self.batch_size,self.num_classes)
        start_time = time.time()
        for i in range(num_iterations):
            print "validating:", i
            #get batch
            x_batch, y_true_batch = val_ext.get_random_batch_balanced()
            #set feed_dict
            feed_dict_train = {self.x: x_batch,self.y_true: y_true_batch}
            #run session
            _,acc,loss = self.sess.run([self.y_pred, self.accuracy,self.cost], feed_dict=feed_dict_train)
            print 'acc: ', acc*100, 'loss: ',loss
        end_time = time.time()
        time_dif = end_time - start_time
        print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


    def test(self):
        test_data = load_test()
        start_time = time.time()
        #open csv
        out = csv.writer(open("test.csv","w"), delimiter=',')
        #create header csv
        out.writerow(['image_name','Type_1','Type_2','Type_3'])
        print test_data.images_count[0]
        for i in range(int(test_data.images_count[0])):
            print "testing:", i
            #get batch
            x_in,name = test_data.get_sample(0,i)
            #set feed_dict
            feed_dict_train = {self.x: [x_in]}
            #run session
            output = self.sess.run([self.y_pred], feed_dict=feed_dict_train)

            #add ouput to csb
            probs =output[0][0].tolist()
            probs[:0] = [name[len('./data/test/'):]]
            print probs
            out.writerow(probs)
        end_time = time.time()
        time_dif = end_time - start_time
        print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


    def save(self, filename):
        self.saver.save(self.sess, filename)

    def load(self, filename):
        self.saver.restore(self.sess, filename)
