from classes import *

class BlockStreamGenerator:

    def __init__(self, BodyScan, SupervisedClassifier, threshold=2.0931642e-05, blockSize=8):
        self.threshold = 2.0931642e-05
        self.blockSize = blockSize
        self.shift = int(blockSize / 2)
        self.bs = BodyScan
        self.sc = SupervisedClassifier

    def generateSegmentStream(self):
        results = [] # final list of all blocks

        # returns 17 2D planes corresponding with body segments
        body_segment_matrix = self.bs.extract_segment_blocks()
        
        # cutting off the filename and just getting the individual image's label
        individual_name = self.bs.filepath.split('/')[-1:][0].split('.')[0]
        # this returns a list of the form [[zone, probability], [], ...] with probability being a binary 1 for threat or 0 for no threat
        threats = self.sc.get_specific_threat_list(individual_name)   

        for b in range(1,16):
            results.append(Block( body_segment_matrix[b][:,:,:], b, threats[b][1], individual_name, 0))

        return results

    def generateStream(self):
        results = [] # final list of all blocks

        # returns 17 2D planes corresponding with body segments
        body_segment_matrix = self.bs.extract_segment_blocks()
        
        # cutting off the filename and just getting the individual image's label
        individual_name = self.bs.filepath.split('/')[-1:][0].split('.')[0]
        
        # this returns a list of the form [[zone, probability], [], ...] with probability being a binary 1 for threat or 0 for no threat
        threats = self.sc.get_specific_threat_list(individual_name)
        #threat_zones = self.sc.get_precise_threat_list()

        # for each 2D body segment iterate through the third dimension until we hit a piece of the 
            # body that matches the threshold we've designated
        for b in range(0,16):
            threat_region = self.sc.get_precise_threat_from_segment(individual_name, b)
            print(threat_region)
            for x in range(0, len(body_segment_matrix[b]), self.shift):
                for z in range(0, len(body_segment_matrix[b][x]), self.shift):
                    for y in range(0, len(body_segment_matrix[b][x][z])):
                        if body_segment_matrix[b][x][z][y] >= self.threshold:
                            is_threat = self.blockInThreatZone(x,y,z, self.bs, threat_region)
                            results.append(Block( body_segment_matrix[b][ x-self.shift:x+self.shift, z-self.shift:z+self.shift, y-self.shift:y+self.shift], b, is_threat, individual_name, self.blockSize))
                            #results.append(Block( body_segment_matrix[b][ x-self.shift:x+self.shift, z-self.shift:z+self.shift, y-self.shift:y+self.shift], b, threats[b][1], individual_name, self.blockSize))
                            break
                    for y in range(len(body_segment_matrix[b][x][z])-1, 0, -1):
                        if(body_segment_matrix[b][x][z][y] >= self.threshold):
                            is_threat = self.blockInThreatZone(x,y,z, self.bs, threat_region)
                            results.append(Block( body_segment_matrix[b][ x-self.shift:x+self.shift, z-self.shift:z+self.shift, y-self.shift:y+self.shift], b, is_threat, individual_name, self.blockSize))
                            #results.append(Block( body_segment_matrix[b][ x-self.shift:x+self.shift, z-self.shift:z+self.shift, y-self.shift:y+self.shift], b, threats[b][1], individual_name, self.blockSize))
                            break
        return results

    def blockInThreatZone(self, x, y, z, blocksize, threat_region):
        # currently we've defined only 1 threat per segment - hopefully this is okay. index our list to get the
        # exact region of the threat
        #threat_zone = threat_zones[segment_num]
        
        # IMPORTANT NOTE: make sure these are using the same axis type
        in_x_region = (threat_region[0][0] < x) & (threat_region[0][1] > x)
        in_y_region = (threat_region[1][0] < y) & (threat_region[1][1] > y)
        in_z_region = (threat_region[2][0] < z) & (threat_region[2][1] > z)

        return in_x_region & in_y_region & in_z_region

class Block:
    def __init__(self, data, region, threat, name, n):
        self.data = data
        self.region = region
        self.threat = threat
        self.name = name
        self.n = n
