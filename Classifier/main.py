from model import simpleModel
from load_data import *

classes = 3
optimize_itt = 500
validation_itt = 300
model = simpleModel(num_classes=classes)



#load checkpoint
model.load('checkpoints/1133531/c-3-itt-500')

# optimize
# model.optimize(optimize_itt,classes,save=True)

# validate
# model.validation(validation_itt)

#test
model.test()
