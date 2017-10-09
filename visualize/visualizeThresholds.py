from classes import *
import matplotlib.pyplot as plt
from skimage import data
from skimage import filters
from skimage import exposure
import numpy as np
import matplotlib.pyplot as plt
from time import time

class ThresholdVisualizer:
	def __init__(self, BodyScan, SupervisedClassifier, scanId):
		self.bs = BodyScan
		self.sc = SupervisedClassifier
		self.scanId = scanId

	def OtsuThresholding(self , data):
		""" Plot Otsu Threshold and return threshold value"""

		val = filters.threshold_otsu(data)

		hist, bins_center = exposure.histogram(data)

		plt.figure(figsize=(9, 4))
		plt.subplot(131)
		plt.imshow(data, cmap='gray', interpolation='nearest')
		plt.axis('off')
		plt.subplot(132)
		plt.imshow(data < val, cmap='gray', interpolation='nearest')
		plt.axis('off')
		plt.subplot(133)
		plt.plot(bins_center, hist, lw=2)
		plt.axvline(val, color='k', ls='--')

		plt.tight_layout()
		plt.show()

		return val

	def ShowHistogram(self,data):
		hist, bins_center = exposure.histogram(data)

		plt.subplot(133)
		plt.figure(figsize=(9, 4))
		plt.plot(bins_center, hist, lw=2)

		plt.tight_layout()
		plt.show()

	def PlotOverlayedHistogram(self):
		# Plot histogram of values for each region - overlayed
		for i in range(0,17):
			data = segments[i]
			hist, bins_center = exposure.histogram(data)
			plt.plot(bins_center, hist, lw = 2)

		plt.show()
	
	def FullRegionHistogram(self):

		segments = self.bs.extract_segment_blocks()
	
		fig, axarr = plt.subplots(nrows=5, ncols=4, figsize=(10,5))

		i = 0

		threatList = self.sc.get_specific_threat_list(self.scanId)

		for row in range(5):
			for col in range(4):
				if(i<17):
					data = segments[i]
					hist, bins_center = exposure.histogram(data)

					print 'hist: ', hist
					print 'bins_center: ', bins_center
					#hist, bins_center = exposure.equalize_adapthist(data)
					# Plot the histograms according to whether or not the region is a threat
					if(threatList[i][1]==1):
						axarr[row, col].plot(bins_center, hist, lw = 2, color = 'red')
					else:
						axarr[row, col].plot(bins_center, hist, lw = 2, color = 'blue')

					axarr[row, col].set_title('Region ' + str(i+1))
					i += 1

		plt.show()

		# Extract maximum values
		for i in range(0,16):
			print(str(i), np.amax(segments[i]), threatList[i][1])

		#data = self.bs.get_single_image(100) # Hardcoded to Otsu Threshold for a single slice
		#data = self.collapse_3D_to_2D(segments[0])

		#self.OtsuThresholding(data)
		#for segment in segments:

	def RemoveBackground(self,data3D, threshold):

		filteredData = []

		for x in range(0, len(data3D)):
			for y in range(0, len(data3D[0])):
				for z in range(0, len(data3D[0][0])):
					if(data3D[x][y][z]>threshold):
						filteredData.append(data3D[x][y][z])

		return filteredData

	def NoBackgroundRegionHistogram(self):
		segments = self.bs.extract_segment_blocks()
		backgroundValue = .001
		i = 0

		threatList = self.sc.get_specific_threat_list(self.scanId)
		fig, axarr = plt.subplots(nrows=5, ncols=4, figsize=(10,5))

		#print data
		#N, bins, patches = plt.hist(data, bins=100)
		#plt.show()

		for row in range(5):
			for col in range(4):
				if(i<17):
					#averageValue = np.mean(segments[i])
					data = self.RemoveBackground(segments[i], backgroundValue)
					N, bins, patches = axarr[row,col].hist(data, bins=50)
					print 'done computing region ', str(i+1)
					"""
					print 'hist: ', hist
					print 'bins_center: ', bins_center
					#hist, bins_center = exposure.equalize_adapthist(data)
					# Plot the histograms according to whether or not the region is a threat
					if(threatList[i][1]==1):
						axarr[row, col].plot(bins_center, hist, lw = 2, color = 'red')
					else:
						axarr[row, col].plot(bins_center, hist, lw = 2, color = 'blue')

					axarr[row, col].set_title('Region ' + str(i+1))
					"""
					i += 1

		plt.show()

