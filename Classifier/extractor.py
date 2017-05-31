from numpy.random import choice
from random import randint
import numpy as np
from PIL import Image

class SampleExtractor:
    def __init__(self, imgs_src):

        # images sources
        self.images_src = imgs_src

        # images count
        self.images_count = np.zeros(len(self.images_src))
        print (self.images_count)
        # images list
        self.images = []

        #load images, i
        for typ,imgs in enumerate(self.images_src):
            #set images count for typ
            count = len(imgs)
            print count
            self.images_count[typ] = count
            # open images
            self.images.append(np.zeros([count, 200, 200,3]))
            for i, img, in enumerate(imgs):
                #load image i of typ
                self.images[typ][i] = np.asarray(Image.open(img))/255.0
            print '\tshape type',typ,self.images[typ].shape

    def get_random_sample_from_class(self, label):
        X = self.images[label][randint(0,self.images_count[label]-1)]
        Y = np.array([int(i == label) for i in range(len(self.images))])
        return X, Y

    def get_sample(self,typ,i):
        return self.images[typ][i], self.images_src[typ][i]


class BatchExtractor:

    def __init__(self, sample_extractor, batch_size,types):
        self.batch_size = batch_size
        self.sample_extractor = sample_extractor
        self.types = types

    def get_random_batch_balanced(self):
        se = self.sample_extractor

        X_batch = []
        Y_batch = []

        m = self.batch_size/self.types
        rest = self.batch_size%self.types
        typ= -1

        for i in range(self.batch_size):
            if i%m==0 and i < self.batch_size-rest: typ+=1
            X, Y = se.get_random_sample_from_class(typ)
            X_batch.append(X)
            Y_batch.append(Y)

        return X_batch, Y_batch
