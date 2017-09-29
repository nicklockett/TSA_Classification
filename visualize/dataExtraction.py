from classes import *

class BlockStreamGenerator:

    def __init__(self, BodyScan, SupervisedClassifier, threshold=.5, blockSize=8):
        self.threshold = 2.0931642e-05
        self.blockSize = blockSize
        self.shift = int(blockSize / 2)
        self.bs = BodyScan
        self.sc = SupervisedClassifier

    def generateStream(self):
        results = [] # final list of all blocks

        # returns 17 2D planes corresponding with body segments
        body_segment_matrix = self.bs.extract_segment_blocks()
        
        # cutting off the filename and just getting the individual image's label
        individual_name = self.bs.filepath.split('/')[-1:][0].split('.')[0]
        # this returns a list of the form [[zone, probability], [], ...] with probability being a binary 1 for threat or 0 for no threat
        threats = self.sc.get_specific_threat_list(individual_name)

		# for each 2D body segment iterate through the third dimension until we hit a piece of the 
			# body that matches the threshold we've designated
		for b in range(0,16):
			for x in range(0, len(body_segment_matrix[b]), self.shift):
				for z in range(0, len(body_segment_matrix[b][x]), self.shift):
					for y in range(0, len(body_segment_matrix[b][x][z])):
						if body_segment_matrix[b][x][z][y] >= self.threshold:
							results.append(Block( body_segment_matrix[b][ x-self.shift:x+self.shift, z-self.shift:z+self.shift, y-self.shift:y+self.shift], b, threats[b][1], individual_name, self.blockSize))
							break
					for y in range(len(body_segment_matrix[b][x][z])-1, 0, -1):
						if(body_segment_matrix[b][x][z][y] >= self.threshold):
							results.append(Block( body_segment_matrix[b][ x-self.shift:x+self.shift, z-self.shift:z+self.shift, y-self.shift:y+self.shift], b, threats[b][1], individual_name, self.blockSize))
							break
		return results


class Block:
	def __init__(self, data, region, threat, name, n):
		self.data = data
		self.region = region
		self.threat = threat
		self.name = name
		self.n = n
