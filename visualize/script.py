from classes import *
from dataExtraction import *

bs = BodyScan("../data/fdb996a779e5d65d043eaa160ec2f09f.a3d")
sc = SupervisedClassifier('../../stage1_labels.csv')
bsg = BlockStreamGenerator(bs, sc)
block_list = bsg.generateStream()
print(block_list[0]) 
print(block_list[0].region)
print(block_list[0].threat)
print(block_list[0].n)