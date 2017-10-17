import sys
sys.path.insert(0, "../visualize/classes.py")
from classes import *
from cv2 import *

class ASF():
	def __init__ (self, file_path):
		bs = BodyScan(file_path)
		self.img_data = bs.img_data
	

	def feature_detection(self):
		features = []
		for 2d in self.img_data:
			#pre processing here with openCV

			# call to other methods to determine results

		return features


	def detect_continuity(self, processedVersionOf2D):

	def line_of_symmetry_deviation(self, ppvo3d):

	def deviation_from_elipse(self, ppvo2d):

	def deviation_from_other_circles(self, ppvo2d):


