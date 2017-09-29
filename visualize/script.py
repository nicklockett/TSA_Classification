from classes import *
from dataExtraction import *
from random import shuffle
from sklearn import svm

# need to ingest and iterate through multiple bodies

bs = BodyScan("../data/fdb996a779e5d65d043eaa160ec2f09f.a3d")
sc = SupervisedClassifier('../../stage1_labels.csv')
bsg = BlockStreamGenerator(bs, sc)
block_list = bsg.generateStream()



