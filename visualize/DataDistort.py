from random import randint
from PIL import Image
import numpy as np

class DataDistort:
    def random_shift_projection(self, projection, width, height):
        shift_limit = 0.15;

        width_shift = randint(-width*shift_limit, width*shift_limit)
        height_shift =randint(-height*shift_limit, height*shift_limit)

        shifted_projection = [[0 for i in xrange(width)] for j in xrange(height)]

        for i in xrange(width):
            for j in xrange(height):
                val = 0
                if (i + width_shift < width and i + width_shift >= 0) and (j + height_shift < height and j + height_shift >=0):
                    val = projection[i][j]
                
                shifted_projection[i][j] = val
        return shifted_projection


    def random_shift_3d(self, data, width, height, depth):
        shift_limit = 0.15;

        width_shift = randint(-width*shift_limit, width*shift_limit)
        height_shift =randint(-height*shift_limit, height*shift_limit)
        depth_shift =randint(-depth*shift_limit, depth*shift_limit)

        shifted_data = [[[0 for i in xrange(width)] for j in xrange(height)] for k in xrange(depth)]

        for i in xrange(width):
            for j in xrange(height):
                for k in xrange(depth):
                    val = 0
                    i_bounds = (i + width_shift < width and i + width_shift >=0)
                    j_bounds = (j + height_shift < height and j + height_shift >=0)
                    k_bounds = (k + depth_shift < depth and k + depth_shift >=0)
                    if i_bounds and j_bounds and k_bounds:
                        val = data[i + width_shift][j + height_shift][k + depth_shift]
                    
                    shifted_data[i][j][k] = val
        return shifted_data

    def random_rotation_projection(self, projection, rotation):
        w_r = randint(-rotation, rotation)
        im = Image.fromarray(projection)
        imR = im.rotate(w_r)
        return np.array(imR)

    def nrr(self, projection, rotation):
        im = Image.fromarray(projection)
        imR = im.rotate(rotation)
        return np.array(imR)

    def random_rotation_3d(self, data, rotation, width, height, depth):
        w_r = randint(-rotation, rotation);
        h_r = randint(-rotation, rotation);
        d_r = randint(-rotation, rotation);
        ans = np.empty([width, height, depth])

        for i in xrange(width):
            ans[i,:,:] = self.nrr(data[i,:,:], w_r)
        for i in xrange(height):
            ans[:,i,:] = self.nrr(data[:,i,:], h_r)
        for i in xrange(depth):
            ans[:,i,:] = self.nrr(data[:,:,i],d_r)
        return ans