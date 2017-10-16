from classes import *
from dataExtraction import *

filepath = "../data/e4e72e96bffb21e55381667911281616.a3d"
labels_fp = "../data/stage1_labels.csv"

dcnn = DeepCNN(labels_fp)
bs = BodyScan(filepath)
bsg = BlockStreamGenerator(bs, dcnn)
blocks = bsg.generateStream()
dcnn.train(blocks)
