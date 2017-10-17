from classes import *
from dataExtraction import *
from visualizeThresholds import *
from random import shuffle
from sklearn import svm
import matplotlib.backends.backend_pdf


scanIds = ["0043db5e8c819bffc15261b1f1ac5e42","011516ab0eca7cad7f5257672ddde70e","0240c8f1e89e855dcd8f1fa6b1e2b944","04b32b70b4ab15cad85d43e3b5359239","052117021fc1396db6bae78ffe923ee4","05aea64cd02c88b94d0663be46f4d2bf","07d04f2ba71419b0d7228f2c50c14318","091cfe2e108e277d82497ee1307f424c","098f5cfcf6faefd3011a94719cb03dc5","0d10b14405f0443be67a75554da778a0","0e34d284cb190a79e3316d1926061bc3","0f47335091ce43a8025ebd2076630dfd","0fdad88d401b09d417ffbc490640d9e2","1020e08af89e2f679f27b4630e55f798"]
sc = SupervisedClassifier('../../stage1_labels.csv')
pdf = matplotlib.backends.backend_pdf.PdfPages("scanAnalysis1.pdf")

for scanId in scanIds:
	threatList = sc.get_specific_threat_list(scanId)
	filepath = "data/a3d/" + scanId + ".a3d"
	bs = BodyScan(filepath)

	threatString = 'Threats at: '
	# Create string analysis of threats
	for threat in threatList:
		if threat[1] == 1:
			threatString += str(threat[0])
			threatString += ', '

	# Plot the body image
	body_data = bs.compress_along_y_z(bs.img_data)
	plt.ion()
	body_fig = plt.figure()
	plt.imshow(body_data)
	body_fig.suptitle(scanId)
	body_fig.text(.5,.5,threatString)
	pdf.savefig(body_fig)
	plt.close(body_fig)

	# Plot slices of the body with contours drawn
	for sliceNum in range(0,600,50):
		slice_fig = bs.find_and_visualize_contours_for_slice(sliceNum)
		pdf.savefig(slice_fig)
		plt.close(slice_fig)

	print(scanId, ' done!')

pdf.close()