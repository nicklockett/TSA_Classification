from classes import *
from dataExtraction import *
from random import shuffle
from sklearn import svm
from ImageSave import *

sc = SupervisedClassifier('../../stage1_labels.csv')
im = ImageSaver(sc)
imagesSaved = im.save_images_to_folder(channels = 3, block_size = 58, resize = -1, segmentNumber = -100, image_filepath = "../../../rec/data/PSRC/Data/stage1/a3d/", nii_filepath = "data/Batch_2D_warp_labels/")
im.load_images_and_save_as_arrays(resize = 224, max_folder = "generated_blocks/block_size_58/max/", sum_folder = "generated_blocks/block_size_58/sum/", var_folder= "generated_blocks/block_size_58/var/")
# need to ingest and iterate through multiple bodies
#fileId = "5e429137784baf5646211dcc8c16ca51"
#bs = BodyScan("data/a3d/"+fileId+".a3d")
#bs.generate_warped_2D_segmentation(fileId)
#sc = SupervisedClassifier('../../stage1_labels.csv')
#threat_range = sc.get_precise_threat_from_segment("0","0")
#print threat_range
#bsg = BlockStreamGenerator(bs, sc, blockSize = 56)
#print 'about to view threat labels'
#bsg.viewThreatLabels()
#block_list = bsg.generate2DBlockStreamHandLabeled()
#print 'done with stream generation'
#bs.plot_3d_from_blocks(block_list)
#for block in block_list:
#	print str(block)

