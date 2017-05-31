import os
import tensorflow as tf

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

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

def get_train_data(data_dir):
    types = [1,2,3]
    print types[0]
    tra_imgs_all = []
    for t in types:
        tra_img_dir = os.path.join(data_dir,'train','Type_'+str(t))
        tra_imgs_all.append(get_file_list(tra_img_dir, 'jpg'))
    return tra_imgs_all
