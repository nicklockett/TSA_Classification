from classes import *
from dataExtraction import *
from random import shuffle
from sklearn import svm

# need to ingest and iterate through multiple bodies

bs = BodyScan("data/a3d/0043db5e8c819bffc15261b1f1ac5e42.a3d")
sc = SupervisedClassifier('../../stage1_labels.csv')
sc.get_precise_threat_from_segment("0","0")
bsg = BlockStreamGenerator(bs, sc)
block_list = bsg.generateStream()



