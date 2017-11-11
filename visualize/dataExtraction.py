from classes import *
import numpy as np
import math
import matplotlib.pyplot as plot
import mpl_toolkits.mplot3d.axes3d as axes3d
import scipy.misc

class BlockStreamGenerator:

    def __init__(self, BodyScan, SupervisedClassifier, threshold=2.0931642e-05, blockSize=8):
        self.threshold = threshold
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

    def cube_marginals(self, cube, normalize=False):
        c_fcn = np.mean if normalize else np.sum
        xy = c_fcn(cube, axis=0)
        xz = c_fcn(cube, axis=1)
        yz = c_fcn(cube, axis=2)
        return(xy,xz,yz)

    def plotcube(self, cube,x=None,y=None,z=None,normalize=False,plot_front=False):
        """Use contourf to plot cube marginals"""
        (Z,Y,X) = cube.shape
        (xy,xz,yz) = self.cube_marginals(cube,normalize=normalize)
        if x == None: x = np.arange(X)
        if y == None: y = np.arange(Y)
        if z == None: z = np.arange(Z)

        fig = plot.figure()
        ax = fig.gca(projection='3d')

        # draw edge marginal surfaces
        offsets = (Z-1,0,X-1) if plot_front else (0, Y-1, 0)
        cset = ax.contourf(x[None,:].repeat(Y,axis=0), y[:,None].repeat(X,axis=1), xy, zdir='z', offset=offsets[0], cmap=plot.cm.coolwarm, alpha=0.75)
        cset = ax.contourf(x[None,:].repeat(Z,axis=0), xz, z[:,None].repeat(X,axis=1), zdir='y', offset=offsets[1], cmap=plot.cm.coolwarm, alpha=0.75)
        cset = ax.contourf(yz, y[None,:].repeat(Z,axis=0), z[:,None].repeat(Y,axis=1), zdir='x', offset=offsets[2], cmap=plot.cm.coolwarm, alpha=0.75)

        # draw wire cube to aid visualization
        ax.plot([0,X-1,X-1,0,0],[0,0,Y-1,Y-1,0],[0,0,0,0,0],'k-')
        ax.plot([0,X-1,X-1,0,0],[0,0,Y-1,Y-1,0],[Z-1,Z-1,Z-1,Z-1,Z-1],'k-')
        ax.plot([0,0],[0,0],[0,Z-1],'k-')
        ax.plot([X-1,X-1],[0,0],[0,Z-1],'k-')
        ax.plot([X-1,X-1],[Y-1,Y-1],[0,Z-1],'k-')
        ax.plot([0,0],[Y-1,Y-1],[0,Z-1],'k-')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plot.show()

    def generateStream(self):
        results = [] # final list of all blocks

        # returns 17 2D planes corresponding with body segments
        body_segment_matrix = self.bs.extract_segment_blocks()
        print('body_segment_matrix: ', body_segment_matrix)

        print('body segment shape:', body_segment_matrix[0].shape)

        self.plotcube(body_segment_matrix[6])

        # cutting off the filename and just getting the individual image's label
        individual_name = self.bs.filepath.split('/')[-1:][0].split('.')[0]
        
        # this returns a list of the form [[zone, probability], [], ...] with probability being a binary 1 for threat or 0 for no threat
        #threats = self.sc.get_specific_threat_list(individual_name)
        #threat_zones = self.sc.get_precise_threat_list()

        # for each 2D body segment iterate through the third dimension until we hit a piece of the 
            # body that matches the threshold we've designated
        for b in range(0,16):
            # Obtain the hand labeled threat region range - will be -1,-1,-1
            threat_region = self.sc.get_precise_threat_from_segment(individual_name, b)
            print threat_region

            print("Region: ", b+1)
            for x in range(0, len(body_segment_matrix[b]), self.shift):
                for z in range(0, len(body_segment_matrix[b][x]), self.shift):
                    for y in range(0, len(body_segment_matrix[b][x][z])):
                        if body_segment_matrix[b][x][z][y] >= self.threshold:
                            # Check to see if this block is in a threat zone
                            is_threat = self.blockInThreatZone(x, y, z, self.bs, threat_region)
                            results.append(Block( body_segment_matrix[b][ x-self.shift:x+self.shift, z-self.shift:z+self.shift, y-self.shift:y+self.shift], b, is_threat, individual_name, self.blockSize))
                            #results.append(Block( body_segment_matrix[b][ x-self.shift:x+self.shift, z-self.shift:z+self.shift, y-self.shift:y+self.shift], b, threats[b][1], individual_name, self.blockSize))
                            break
                    for y in range(len(body_segment_matrix[b][x][z])-1, 0, -1):
                        if(body_segment_matrix[b][x][z][y] >= self.threshold):
                            # Check to see if this block is in a threat zone
                            is_threat = self.blockInThreatZone(x,y,z, self.bs, threat_region)
                            results.append(Block( body_segment_matrix[b][ x-self.shift:x+self.shift, z-self.shift:z+self.shift, y-self.shift:y+self.shift], b, is_threat, individual_name, self.blockSize))
                            #results.append(Block( body_segment_matrix[b][ x-self.shift:x+self.shift, z-self.shift:z+self.shift, y-self.shift:y+self.shift], b, threats[b][1], individual_name, self.blockSize))
                            break
        return results

    def generate3DBlockStreamHandLabeled(self):
        results = []
        individual_id = self.bs.filepath.split('/')[-1:][0].split('.')[0]
        threat_cubes = self.sc.get_threatcubes(individual_id)
        full_body_data = self.bs.img_data
        segmented_data = self.bs.generate_warped_2D_segmentation(individual_id)

        print full_body_data.shape

        for x in range(0, len(full_body_data), self.shift):
            for y in range(0, len(full_body_data[x]), self.shift):
                for z in range(0, len(full_body_data[x][y]), self.shift):
                    if(full_body_data[x][y][z] >= self.threshold):
                        is_threat = self.classifyThreat(x, y, z, threat_cubes)
                        region_label = self.classifyRegion(x,y,z, segmented_data)
                        results.append(Block(full_body_data[ x-self.shift:x+self.shift, z-self.shift:z+self.shift, y-self.shift:y+self.shift], region_label, is_threat, individual_id, self.blockSize))
                        if(is_threat):
                            print(region_label)

        return results

    def generate2DBlockStreamHandLabeled(self):
        block_stream = []
        individual_id = self.bs.filepath.split('/')[-1:][0].split('.')[0]
        image_2D = self.bs.flatten_max()
        threat_cubes = self.sc.get_threatcubes(individual_id)
        segmented_data = self.bs.generate_warped_2D_segmentation(individual_id)

        for x in range(0, len(image_2D), self.shift):
            for y in range(0, len(image_2D[x]), self.shift):
                    region_label = self.classifyRegion2D(x, y, segmented_data)
                    if(region_label!=1.0):
                        is_threat = self.classifyThreat2D(x, y, threat_cubes)
                        data = image_2D[x-self.shift:x+self.shift, y-self.shift:y+self.shift]
                        block_stream.append((data, region_label, is_threat))
                        if(is_threat):
                            print(region_label)

        return block_stream

    def generate2DBlockStreamHandLabeled3Channel(self, resize=-1):
        block_stream = []
        individual_id = self.bs.filepath.split('/')[-1:][0].split('.')[0]

        image_Max = self.bs.flatten_max()
        image_Sum = self.bs.flatten_sum()
        image_Var = self.bs.flatten_var()

        threat_cubes = self.sc.get_threatcubes(individual_id)
        print threat_cubes
        segmented_data = self.bs.generate_warped_2D_segmentation(individual_id)

        for x in range(0, len(image_Max), self.shift):
            for y in range(0, len(image_Max[x]), self.shift):
                    region_label = self.classifyRegion2D(x, y, segmented_data)
                    if(region_label!=1.0):
                        is_threat = self.classifyThreat2D(x, y, threat_cubes)

                        Channeled_Data = np.zeros((self.blockSize,self.blockSize, 3))

                        data_channel_1 = image_Max[x-self.shift:x+self.shift, y-self.shift:y+self.shift]
                        data_channel_2 = image_Sum[x-self.shift:x+self.shift, y-self.shift:y+self.shift]
                        data_channel_3 = image_Var[x-self.shift:x+self.shift, y-self.shift:y+self.shift]

                        if(resize!=-1):
                            Channeled_Data = np.zeros((resize,resize, 3))
                            data_channel_1 = scipy.misc.imresize(arr = data_channel_1, size=(resize, resize))
                            data_channel_2 = scipy.misc.imresize(arr = data_channel_2, size=(resize, resize))
                            data_channel_3 = scipy.misc.imresize(arr = data_channel_3, size=(resize, resize))

                        for r in range(0,len(data_channel_1)):
                            for c in range(0,len(data_channel_1[0])):
                                Channeled_Data[r][c][0] = data_channel_1[r][c]
                                Channeled_Data[r][c][1] = data_channel_2[r][c]
                                Channeled_Data[r][c][2] = data_channel_3[r][c]

                        block_stream.append((Channeled_Data, region_label, is_threat))
                        if(is_threat):
                            print(region_label)
        return block_stream

    def classifyRegion2D(self, x, y, segmented_data):
        return segmented_data[660-1-x][y]

    def classifyThreat2D(self, x, y, threat_regions):
        for region in threat_regions:
            if(self.blockInThreatZone2D(x, y, region)):
                return True

        return False 

    def blockInThreatZone2D(self, x, y, threat_region):
        # If this isn't a segment with a threat anyways, return False
        if(threat_region==-1):
            return False

        in_x_region = (threat_region[0][0] < y) & (y < threat_region[0][1])
        in_y_region = (threat_region[1][0] < x) & (x <  threat_region[1][1])

        return in_x_region & in_y_region

    def viewThreatLabels(self):
        individual_id = self.bs.filepath.split('/')[-1:][0].split('.')[0]
        threat_cubes = self.sc.get_threatcubes(individual_id)
        print threat_cubes
        full_body_data = self.bs.img_data
        segmented_data = self.bs.generate_warped_2D_segmentation(individual_id)

        print 'beginning our scan through the data'

        max_val = np.amax(full_body_data)

        for x in range(0, len(full_body_data), self.shift):
            for y in range(0, len(full_body_data[x]), self.shift):
                for z in range(0, len(full_body_data[x][y]), self.shift):
                    region_label = self.classifyRegion(x,y,z, segmented_data)
                    is_threat = self.classifyThreat(x, y, z, threat_cubes)
                    if(is_threat):
                    #if(region_label!=1.0 or is_threat):
                        full_body_data[x][y][z] = max_val
                        print 'is_threat'
                    else:
                        full_body_data[x][y][z] = 0.0

        print 'about to compress_along_x_y'
        #print full_body_data
        #flattened_data = self.bs.compress_along_y_z(full_body_data)
        flattened_data = self.bs.flatten_max()
        print 'done compressing, now graphing'

        plt.figure()
        plt.imshow(flattened_data)
        plt.show()

        print 'done!'

    def classifyRegion(self, x, y, z, segmented_data):
        #print segmented_data.item(x,y)
        return segmented_data[z][x]

    def classifyThreat(self, x, y, z, threat_regions):
        for region in threat_regions:
            if(self.blockInThreatZone(x, y, z, region)):
                return True

        return False

    def blockInThreatZone(self, x, y, z, threat_region):
        # If this isn't a segment with a threat anyways, return False
        if(threat_region==-1):
            return False

        in_x_region = (threat_region[0][0] < x) & (x < threat_region[0][1])
        in_y_region = ((660 - threat_region[1][1]) < z) & (z < (660 - threat_region[1][0]))
        in_z_region = (threat_region[2][0] < y) & (y < threat_region[2][1])

        return in_x_region & in_y_region & in_z_region

class Block:
    def __init__(self, data, region, threat, name, n):
        self.data = data
        self.region = region
        self.threat = threat
        self.name = name
        self.n = n

    def __str__(self):
        return str(self.data.shape) + ", " + str(self.region) + ", " + str(self.threat)
