from classes import *

class BlockStreamGenerator:

	def __init__(self, BodyScan, threshold=.5, blockSize=8):
		self.threshold = .5
		self.blockSize = blockSize
		self.shift = blockSize/2
		self.bs = BodyScan

	def generateStream(self):
		results = []
		threats = {}

		Blocks = self.bs.extract_segment_blocks()
		for i in range(0, 16):
			threats[i] = False
			#TODO: Change this to reading in the values

		for b in range(0,16):
			for x in range(0, len(Blocks[b]), self.shift):
				for z in range(0, len(Blocks[b][x]), self.shift):
					for y in range(0, len(Blocks[b][x][z])):
						if Blocks[b][x][z][y] >= self.threshold:
							results.append(Block( Blocks[b][ x-self.shift:x+self.shift, z-self.shift:z+self.shift, y-self.shift:y+self.shift], b, threats[b], self.blockSize))
							break
					for y in range(len(Blocks[b][x][z])-1, 0, -1):
						if(Blocks[b][x][z][y] >= self.threshold):
							results.append(Block( Blocks[b][ x-self.shift:x+self.shift, z-self.shift:z+self.shift, y-self.shift:y+self.shift], b, threats[b], self.blockSize))
							break

		return results


class Block:
	def __init__(self, data, region, threat, n):
		self.data = data
		self.region = region
		self.threat = threat
		self.n = n
