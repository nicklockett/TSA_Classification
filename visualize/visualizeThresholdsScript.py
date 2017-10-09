from classes import *
from dataExtraction import *
from visualizeThresholds import *
from random import shuffle
from sklearn import svm

# need to ingest and iterate through multiple bodies
scanId = "0043db5e8c819bffc15261b1f1ac5e42"
filepath = "data/a3d/" + scanId + ".a3d"
bs = BodyScan(filepath)
sc = SupervisedClassifier('../../stage1_labels.csv')
tv = ThresholdVisualizer(bs, sc, scanId)
threatList = sc.get_specific_threat_list(scanId)
print "Threats: \n", threatList

#tv.FullRegionThresholding()
tv.NoBackgroundRegionHistogram()
#bsg = BlockStreamGenerator(bs, sc)
#block_list = bsg.generateStream()
