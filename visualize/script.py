from classes import *
from dataExtraction import *
from random import shuffle
from sklearn import svm

# need to ingest and iterate through multiple bodies
fileId = "0e34d284cb190a79e3316d1926061bc3"
bs = BodyScan("data/a3d/"+fileId+".a3d")
#bs.generate_warped_2D_segmentation(fileId)
sc = SupervisedClassifier('../../stage1_labels.csv')
#threat_range = sc.get_precise_threat_from_segment("0","0")
#print threat_range
bsg = BlockStreamGenerator(bs, sc, blockSize = 8)
print 'about to view threat labels'
bsg.viewThreatLabels()
#block_list = bsg.generate2DBlockStreamHandLabeled()
print 'done with stream generation'
#bs.plot_3d_from_blocks(block_list)
#for block in block_list:
#	print str(block)

