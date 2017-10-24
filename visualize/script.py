from classes import *
from dataExtraction import *
from random import shuffle
from sklearn import svm

# need to ingest and iterate through multiple bodies

bs = BodyScan("data/a3d/0043db5e8c819bffc15261b1f1ac5e42.a3d")
sc = SupervisedClassifier('../../stage1_labels.csv')
threat_range = sc.get_precise_threat_from_segment("0","0")
print threat_range
bsg = BlockStreamGenerator(bs, sc, blockSize = 40)
block_list = bsg.generateStreamHandLabeled()
#bs.plot_3d_from_blocks(block_list)
#for block in block_list:
#	print str(block)

