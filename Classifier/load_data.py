from extractor import *
from utils import *
import numpy as np

def load_train_data(batch_size,types):
    if types == 3: return load_train_data3(batch_size,types)
    if types == 2: return load_train_data2(batch_size,types)
    print 'error'

##!!!!redundantie!!!!!
def load_train_data3(batch_size,types):
    data_dir = "./data/"
    type1_dir = "cropType1"
    type2_dir = "cropType2"
    type3_dir = "cropType3"

    type1_src = get_file_list(data_dir+type1_dir, 'jpg')
    type2_src = get_file_list(data_dir+type2_dir, 'jpg')
    type3_src = get_file_list(data_dir+type3_dir, 'jpg')

    print 'type1 images count: ', len(type1_src)
    print 'type2 images count: ', len(type2_src)
    print 'type3 images count: ', len(type3_src)

    val = int(len(type1_src)*0.3)
    type1_train= type1_src[val:]
    type1_val = type1_src[:val]

    val = int(len(type2_src)*0.3)
    type2_train= type2_src[val:]
    type2_val = type2_src[:val]

    val = int(len(type3_src)*0.3)
    type3_train= type3_src[val:]
    type3_val = type3_src[:val]

    print 'create train batch extractor'
    train_batch = BatchExtractor(SampleExtractor([type1_train,type2_train,type3_train]),batch_size,types)
    print 'create validation batch extractor'
    val_batch = BatchExtractor(SampleExtractor([type1_val,type2_val,type3_val]),batch_size,types)

    return train_batch, val_batch

def load_train_data2(batch_size,types):
    data_dir = "./data/"
    type1_dir = "cropType1"
    type3_dir = "cropType3"

    type1_src = get_file_list(data_dir+type1_dir, 'jpg')
    type3_src = get_file_list(data_dir+type3_dir, 'jpg')

    print 'type1 images count: ', len(type1_src)
    print 'type3 images count: ', len(type3_src)

    val = int(len(type1_src)*0.3)
    type1_train= type1_src[val:]
    type1_val = type1_src[:val]

    val = int(len(type3_src)*0.3)
    type3_train= type3_src[val:]
    type3_val = type3_src[:val]

    print 'create train batch extractor'
    train_batch = BatchExtractor(SampleExtractor([type1_train,type3_train]),batch_size,types)
    print 'create validation batch extractor'
    val_batch = BatchExtractor(SampleExtractor([type1_val,type3_val]),batch_size,types)

    return train_batch, val_batch

def load_test():
    test_src = get_test_src("./data/")
    return SampleExtractor([test_src])
