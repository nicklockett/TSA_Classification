from __future__ import print_function
from __future__ import division
from abc import ABCMeta
from matplotlib import pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import cv2
import matplotlib
import numpy as np
import itertools
import os
import os.path
import pandas as pd
import scipy.stats as stats
import png
import re
import random
import seaborn as sns
from sklearn import svm, metrics, tree, naive_bayes
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.decomposition import PCA
from nibabel.testing import data_path
import nibabel as nib
from constants import *


class Block(object):
    def __init__(self, data, coords, threat=0):
        self.data = data
        self.coords = coords
        self.threat = threat

    def __eq__ (self, other):
        return self.coords[0] == other.coords[0] and \
            self.coords[1] == other.coords[1] and \
            self.coords[2] == other.coords[2]

    def __hash__(self):
        return self.coords[0] + self.coords[1]*1000 + self.coords[2]*1000*1000


class BodyScan(object):
    """
    Main class for body scan data.
    """
    CONTOUR_THRESH_LB = 70
    CONTOUR_THRESH_UB = 255

    def __init__(self, filepath):
        """
        Initializes the BodyScan object using the file specified. Accepts
        .aps, .aps3d, .a3d, or ahi files
        """
        self.filepath = filepath
        self.header = self.read_header()
        self.person_id = re.search(r"\/(\w+)\.", filepath).group(1)
        self.img_data, self.imag = self.read_img_data()  # real and imaginary

    def read_header(self):
        """
        Takes an aps file and creates a dict of the data
        and returns all of the fields in the header
        """
        infile = self.filepath

        # declare dictionary
        h = dict()

        # read the aps file
        with open(infile, 'r+b') as fid:
            h['filename'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 20))
            h['parent_filename'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 20))
            h['comments1'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 80))
            h['comments2'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 80))
            h['energy_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
            h['config_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
            h['file_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
            h['trans_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
            h['scan_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
            h['data_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
            h['date_modified'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 16))
            h['frequency'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['mat_velocity'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['num_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
            h['num_polarization_channels'] = np.fromfile(fid, dtype = np.int16, count = 1)
            h['spare00'] = np.fromfile(fid, dtype = np.int16, count = 1)
            h['adc_min_voltage'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['adc_max_voltage'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['band_width'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['spare01'] = np.fromfile(fid, dtype = np.int16, count = 5)
            h['polarization_type'] = np.fromfile(fid, dtype = np.int16, count = 4)
            h['record_header_size'] = np.fromfile(fid, dtype = np.int16, count = 1)
            h['word_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
            h['word_precision'] = np.fromfile(fid, dtype = np.int16, count = 1)
            h['min_data_value'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['max_data_value'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['avg_data_value'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['data_scale_factor'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['data_units'] = np.fromfile(fid, dtype = np.int16, count = 1)
            h['surf_removal'] = np.fromfile(fid, dtype = np.uint16, count = 1)
            h['edge_weighting'] = np.fromfile(fid, dtype = np.uint16, count = 1)
            h['x_units'] = np.fromfile(fid, dtype = np.uint16, count = 1)
            h['y_units'] = np.fromfile(fid, dtype = np.uint16, count = 1)
            h['z_units'] = np.fromfile(fid, dtype = np.uint16, count = 1)
            h['t_units'] = np.fromfile(fid, dtype = np.uint16, count = 1)
            h['spare02'] = np.fromfile(fid, dtype = np.int16, count = 1)
            h['x_return_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['y_return_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['z_return_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['scan_orientation'] = np.fromfile(fid, dtype = np.int16, count = 1)
            h['scan_direction'] = np.fromfile(fid, dtype = np.int16, count = 1)
            h['data_storage_order'] = np.fromfile(fid, dtype = np.int16, count = 1)
            h['scanner_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
            h['x_inc'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['y_inc'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['z_inc'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['t_inc'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['num_x_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
            h['num_y_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
            h['num_z_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
            h['num_t_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
            h['x_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['y_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['z_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['x_acc'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['y_acc'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['z_acc'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['x_motor_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['y_motor_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['z_motor_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['x_encoder_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['y_encoder_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['z_encoder_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['date_processed'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 8))
            h['time_processed'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 8))
            h['depth_recon'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['x_max_travel'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['y_max_travel'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['elevation_offset_angle'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['roll_offset_angle'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['z_max_travel'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['azimuth_offset_angle'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['adc_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
            h['spare06'] = np.fromfile(fid, dtype = np.int16, count = 1)
            h['scanner_radius'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['x_offset'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['y_offset'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['z_offset'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['t_delay'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['range_gate_start'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['range_gate_end'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['ahis_software_version'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['spare_end'] = np.fromfile(fid, dtype = np.float32, count = 10)

        return h

    def read_img_data(self):
        """
        Reads .aps, .aps3d, .a3d, or ahi files and returns a stack of images.
        reads and rescales any of the four image types
        """
        # read in header and get dimensions
        infile = self.filepath
        h = self.header
        nx = int(h['num_x_pts'])
        ny = int(h['num_y_pts'])
        nt = int(h['num_t_pts'])

        extension = os.path.splitext(infile)[1]

        with open(infile, 'rb') as fid:
            # skip the header
            fid.seek(512)

            # handle .aps and .a3aps files
            # word_type == 7 is an np.float32, word_type == 4 is np.uint16
            if extension == '.aps' or extension == '.a3daps':
                if(h['word_type']==7):
                    data = np.fromfile(fid, dtype=np.float32, count=nx * ny * nt)
                elif(h['word_type']==4): 
                    data = np.fromfile(fid, dtype=np.uint16, count=nx * ny * nt)

                # scale and reshape the data
                data = data * h['data_scale_factor'] 
                data = data.reshape(nx, ny, nt, order='F').copy()

            # handle .a3d files
            elif extension == '.a3d':
                if(h['word_type']==7):
                    data = np.fromfile(fid, dtype=np.float32, count=nx * ny * nt)
                elif(h['word_type']==4):
                    data = np.fromfile(fid, dtype=np.uint16, count=nx * ny * nt)
                # scale and reshape the data
                data = data * h['data_scale_factor']
                data = data.reshape(nx, nt, ny, order='F').copy()

            # handle .ahi files
            elif extension == '.ahi':
                data = np.fromfile(fid, dtype=np.float32, count=2 * nx * ny * nt)
                data = data.reshape(2, ny, nx, nt, order='F').copy()
                real = data[0,:,:,:].copy()
                imag = data[1,:,:,:].copy()

            if extension != '.ahi':
                return data, None
            else:
                return real, imag

    def plot_image_set(self):
        """
        takes an aps file and shows all 16 90 degree shots
        """
        # read in the aps file, it comes in as shape(512, 620, 16)
        img = self.img_data

        # transpose so that the slice is the first dimension shape(16, 620, 512)
        img = img.transpose()

        # show the graphs
        fig, axarr = plt.subplots(nrows=4, ncols=4, figsize=(10,10))
        i = 0
        for row in range(4):
            for col in range(4):
                resized_img = cv2.resize(img[i], (0,0), fx=0.1, fy=0.1)
                axarr[row, col].imshow(np.flipud(resized_img), cmap=COLORMAP)
                i += 1
        print('Done!')

    def get_contours(self, img_slice):
        """
        Given a slice image in an R2 matrix,
        returns an image with the contours.
        """
        gray_im = self.convert_to_grayscale(img_slice)
        ret, thresh = cv2.threshold(gray_im, self.CONTOUR_THRESH_LB, self.CONTOUR_THRESH_UB, 0) # Can experiment with these boundry values
        im2, contours, hierarchy = cv2.findContours(thresh, 1, 2)
        return im2, contours, hierarchy

    def find_and_visualize_contours_for_slice(self, slice_number):
        # Get initial slice and visualize it in grayscale
        trans_data = self.img_data.transpose()

        # (https://stackoverflow.com/questions/30249053/python-opencv-drawing-errors-after-manipulating-array-with-numpy)
        trans_data_copy = trans_data.copy() #<-- look above for explanation
        gray_im = self.convert_to_grayscale(trans_data_copy[slice_number])
        plt.ion()
        f, (ax1, ax2) = plt.subplots(1, 2)

        ax1.imshow(gray_im)
        ax2.imshow(gray_im)

        im2,contours,hierarchy = self.get_contours(trans_data_copy[slice_number])
        cv2.drawContours(gray_im, contours, -1, (0,255,0), 3)

        # Visualize the min enclosing circles around each of the contours
        # that our function latches onto
        for cnt in contours:
            (x,y),radius = cv2.minEnclosingCircle(cnt)
            center = (int(x),int(y))
            radius = int(radius)
            ax2.imshow(cv2.circle(gray_im,center,radius,(0,255,0),2))

        return f

    #def get_segmented_data():
    #    full_body_data = self.img_data

     #   for x in range(0, len(full_body_data)):
      #      for y in range(0, len(full_body_data[x])):
       #         for z in range(0, len(full_body_data[x][y])):
                    


    def compress_along_y_z(self, full_data):
        print(full_data.shape)

        matrix2D = np.zeros((full_data.shape[2],full_data.shape[1]))
        print (matrix2D.shape)

        for x in range(0, len(full_data)):
            for z in range(0, len(full_data[0][0])):
                value_sum = 0
                for y in range(0, len(full_data[0])):
                    value_sum += full_data[x][y][z]
                matrix2D[len(full_data[0][0])-1-z][x] = value_sum

        print(matrix2D.shape)

        return matrix2D

    def compress_along_x_y(self, full_data):
        print(full_data.shape)

        matrix2D = np.zeros((full_data.shape[0],full_data.shape[1]))
        print (matrix2D.shape)

        for x in range(0, len(full_data)):
            for y in range(0, len(full_data[0])):
                value_sum = 0
                for z in range(0, len(full_data[0][0])):
                    value_sum += full_data[x][y][z]
                matrix2D[x][y] = value_sum

        print(matrix2D.shape)

        return matrix2D

    def toe_to_head_sweep(self):
        """
        performs and visualizes toe_to_head sweep
        """
        data = self.img_data.transpose()
        plt.ion()
        fig = plt.figure()
        for dslice in data:
            plt.clf()
            plt.imshow(dslice, cmap="hot")
            fig.canvas.draw()

    def get_single_image(self, nth_image):
        """
        Returns the nth image from the image stack
        """

        # read in the aps file, it comes in as shape(512, 620, 16)
        img = self.img_data

        # transpose so that the slice is the first dimension shape(16, 620, 512)
        img = img.transpose()

        return np.flipud(img[nth_image])

    def extract_segment_blocks(self):
        """
        Returns a matrix for each body segment
        """
        region_matrices = []
        img = self.img_data
        img_data_trans = img.transpose()
        img_data_trans = np.swapaxes(img_data_trans,1,2)
        img_data_trans = np.flip(img_data_trans, 0)

        # iterate through each segment zone
        for i in range(17):
            region_matrices.append(self.crop(img_data_trans, sector_crop_list[i]))

        return region_matrices

    def convert_to_grayscale(self, img):
        """
        Converts an ATI scan to grayscale and returns an img
        """
        # scale pixel values to grayscale
        base_range = np.amax(img) - np.amin(img)
        rescaled_range = 255 - 0
        img_rescaled = (((img - np.amin(img)) * rescaled_range) / base_range)

        return np.uint8(img_rescaled)

    def spread_spectrum(self, img):
        """
        Applies a histogram equalization transformation and returns a transformed scan.
        """
        img = stats.threshold(img, threshmin=12, newval=0)

        # see http://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img= clahe.apply(img)

        return img

    def roi(self, img, vertices):
        """
        uses vertices to mask the image. Verticies include a set of vertices
        that define the region of interest. It returns the masked image.
        """

        # blank mask
        mask = np.zeros_like(img)

        # fill the mask
        cv2.fillPoly(mask, [vertices], 255)

        # now only show the area that is the mask
        masked = cv2.bitwise_and(img, mask) 

        return masked

    def crop(self, img, crop_list):
        """
        uses vertices to mask the image and returns a cropped image.
        Crop list is an entry with [x, y, width, height]
        """

        x_coord = crop_list[0]
        y_coord = crop_list[1]
        width = crop_list[2]
        height = crop_list[3]
        cropped_img = img[x_coord:x_coord+width, y_coord:y_coord+height]

        return cropped_img

    def flatten_sum(self, thresh=3.5e-04, axis=1):
        """
        Flattens the 3D image to a 2d matrix using sum.
        """
        return np.rot90(np.sum(self.img_data, axis=axis))

    def flatten_max(self, thresh=3.5e-04, axis=1):
        """
        Flattens the 3D image to a 2d matrix using max.
        """
        return np.rot90(np.max(self.img_data, axis=axis))

    def get_filepaths(self, directory):
        """
        retrieves a list of all filepaths from this directory
        """
        output = []

        onlyfiles = [
            f
            for f in os.listdir(os.path.realpath("."))
            if os.path.isfile(os.path.join(os.path.realpath("."), f))
        ]

        for k in onlyfiles:
            output.append(os.path.join(directory, k))

        return output

    def get_a3d_filepaths(self, directory):
        """
        retrieves a list of a3d filepaths from this directory
        """
        output = []

        onlyfiles = self.get_filenames(directory)

        for k in onlyfiles:
            if k.endswith(".a3d"):
                output.append(k)

        return output

    def create_max_projection(self, file_path, axis=1):
        """
        flattens and writes to image
        """
        #fname = "processed/" + self.person_id + "_1.png"
        #matrix = self.flatten_sum(axis=axis)
        #self.write_slice_to_img(matrix, fname)

        fname = file_path + self.person_id + "_projection.png"
        matrix = self.flatten_max(axis=axis)
        self.write_slice_to_img(matrix, fname)
        return(fname)

    def write_slice_to_img(self, slic, filename):
        """
        Given a slice, writes it to a png
        """
        slic = slic / np.max(slic) * 255
        if not filename.endswith(".png"):
            filename += ".png"
        with open(filename, "wb") as f:
            w = png.Writer(512, 660, greyscale=True)
            w.write(f, slic)

    def write_square_slice_to_img(self, slic, filename):
        """
        Given a slice, writes it to a png
        """
        slic = slic / np.max(slic) * 255
        with open(filename, "wb") as f:
            w = png.Writer(512, 512, greyscale=True)
            w.write(f, slic)

    def normalize(self, image):
        """
        Take segmented tsa image and normalize pixel values to be 
        between 0 and 1 and returns a normalized image
        """
        MIN_BOUND = 0.0
        MAX_BOUND = 255.0

        image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
        image[image>1] = 1.
        image[image<0] = 0.
        return image

    def zero_center(self, image):
        """
        Shift normalized image data and move the range so it is 0 c
        entered at the PIXEL_MEAN and returns a zero centered image
        """
        PIXEL_MEAN = 0.014327

        image = image - PIXEL_MEAN
        return image

    def find_nearest_point(self, point, image, num_tries=10000):
        """
        Given a tuple of point, finds kind of the nearest point on the image
        that exists. By that i mean it goes around in squares
        """
        queue = Queue.Queue()

        queue.put(point)

        # xbound
        xb = len(image) - 1

        # ybound
        yb = len(image[0]) - 1

        # initialize visited matrix
        visited = np.zeros((len(image), len(image[0]))) 

        # if > num tries then returns none

        count = 0
        while not queue.empty():
            count += 1
            if count > num_tries:
                return None
            point = queue.get()
            x = point[0]
            y = point[1]

            # add the next points to visit for q
            if (x+1 <= xb) and not visited[x+1][y]:
                queue.put((x+1, y))
            if (x-1 >= 0) and not visited[x-1][y]:
                queue.put((x-1, y))
            if (y+1 <= yb) and not visited[x][y+1]:
                queue.put((x, y+1))
            if (y-1 >= 0) and not visited[x][y-1]:
                queue.put((x, y-1))
            
            # visit current point
            visited[x][y] = 1

            if image[x][y]:
                return (x, y)

        # if we can't find a point, just return none.
        return None

    def find_nearest_point_from_contours(self, point, contours):
        """
        """
        nearest_point = (9999,9999)
        distance = 99999999
        for pt in contours:
            dist = self.get_distance(point, pt)
            if dist < distance:
                nearest_point = pt
                distance = dist

        if distance == 99999999:
            return None
        return nearest_point

    def get_distance(self, pt1, pt2):
        """
        gets the distance between two points as tuples
        """
        return np.sqrt(np.power(pt1[0] - pt2[0], 2) + np.power(pt1[1] - pt2[1], 2))

    def collapse_to_tuple_array(self, contour_list):
        """
        Takes a list of contours and collapes them to a tuple array
        """
        output = []
        for m1 in contour_list:
            for m2 in m1:
                for m3 in m2:
                    output.append((m3[0], m3[1]))

        return sorted(output)

    def get_continuity(self, slice_number, data_cube):
        """
        Given slice number and data cube, which is the 3d body scan
        oriented the correct way, returns the continuity score
        """
        cur = data_cube[slice_number]

        # only try to access if it exists
        if slice_number < len(data_cube):
            nex = data_cube[slice_number+1]
        else:
            # print "reached the end of the thing and nothing to compere to"
            return 0

        # get the images of contours
        cur_contour_img, cur_contours, _ = self.get_contours(cur)
        nex_contour_img, nex_contours, _ = self.get_contours(nex)

        cur_contours = self.collapse_to_tuple_array(cur_contours)
        nex_contours = self.collapse_to_tuple_array(nex_contours)

        tot = 0.0
        count = 0

        k = 0

        for l, k in cur_contours:
            # nearest_point = self.find_nearest_point((k, l), nex_contour_img)
            nearest_point = self.find_nearest_point_from_contours((k, l), nex_contours)
            if nearest_point:
                dist = self.get_distance((k, l), nearest_point)
            else:
                dist = None
            if dist and dist >= 0:
                tot += dist
                count += 1

        if count == 0:
            count = float(0.001)

        return tot / float(count)

    def extract_blocks_from_segment(self, segment_number, threshold=1e-03, block_size=8, shift=1):
        """
        Returns an array of blocks generated around the surface of the
        body in the specified segment
        """
        segments = self.extract_segment_blocks()

        return self.generate_blocks(segments[segment_number], threshold, block_size, shift)

    def generate_blocks(self, body_segment_matrix=None, threshold=1e-04, block_size=8, shift=8, use_max=True):
        """
        Generates blocks from a cropped segment
        """
        if not body_segment_matrix:
            body_segment_matrix = self.img_data

        bsm = body_segment_matrix
        n = block_size
        ds = int(n / 2)

        output = []
        for x in range(ds, len(bsm)-ds, shift):
            lb = ds
            ub = len(bsm[x][ds]-ds-1)
            for z in range(ds, len(bsm[x])-ds, shift):
                for y in range(lb, ub, shift):
                    if not use_max:
                        av = np.average(bsm[x-ds:x+ds, z-ds:z+ds, y-ds:y+ds])
                    else:
                        av = np.max(bsm[x-ds:x+ds, z-ds:z+ds, y-ds:y+ds])
                    if av >= threshold:
                        data = bsm[x-ds:x+ds, z-ds:z+ds, y-ds:y+ds]
                        coords = (x, z, y)
                        block = Block(data, coords)
                        output.append(block)

        return output

    def plot_3d_from_blocks(self, blocks):
        """
        plots 3d of the blocks.
        """
        x = []
        y = []
        z = []
        c = []
        for bl in blocks:
            x.append(bl.coords[0])
            y.append(bl.coords[1])
            z.append(bl.coords[2])
            c.append(np.average(bl.data))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z, c=c, cmap=plt.hot())

        axes = plt.gca()
        axes.set_xlim([0,512])
        axes.set_ylim([0,512])
        axes.set_zlim([0,660])

        plt.show()

    def stochastic_gradient_descent(self, data, dim=2, n=3, alpha=1e-4, max_iter=100, thresh=10, chunk=5):
        # initialize variables
        tot_n = np.power((n + 1), dim)
        weights = np.ones(tot_n) / tot_n
        c_list = []
        alphas = np.array([])
        data_shape = data.shape

        # initialize c_list
        pools = itertools.product(range(n+1), repeat=dim)
        for k in pools:
            c_list.append(np.array(k))

        for c in c_list:
            alph = np.linalg.norm(c)
            if alph == 0:
                alph = alpha
            else:
                alph = alpha / alph
            alphas = np.append(alphas, alph)

        # descend!
        counter = 0
        error = 100.0

        while (counter < max_iter and error > thresh):
            error = 0.0
            for index, s in np.ndenumerate(data):
                if (np.array(index) % chunk).any():
                    continue
                phis = np.array([])
                x = np.ones(dim)
                # normalizing x to 0 < x < 1
                for l in range(dim):
                    x[l] = float(index[l]) / data_shape[l]
                for c in c_list:
                    ret = self.phi(x, c)
                    phis = np.append(phis, ret)

                cur_error = np.dot(weights, phis) - s
                error += np.power(cur_error, 2) / data.size
                weights = weights - phis * alphas * cur_error
            counter += 1
            print("cur weight:", weights)
            print("cur error: ", error)

        return weights, c_list

    def phi(self, x, c):
        """
        returns phi. x must be in [-1, 1]
        """
        return np.cos(np.pi * np.dot(x, c))

    def generate_warped_2D_segmentation(self, fileId):
        #fileId = "fdb996a779e5d65d043eaa160ec2f09f"
        example_file = "data/Batch_2D_warp_labels/" + fileId + "_warp.nii"
        img = nib.load(example_file)
        img_data =  img.get_data()

        data = []
        for row in img_data:
            new_row = []
            for item in row:
                new_row.append(item)
            data.append(new_row)

        final_data = np.array(data)
        final_data = np.flip(final_data, 1)

        return final_data

class SupervisedClassifier(object):
    __metaclass__ = ABCMeta
    """
    This is the wrapper class for the classifier ML logic.
    The methods shared by all classifiers should be in here.
    """
    def __init__(self, labels_filepath):
        """
        Initializes labels_filepath
        """
        self.labels_filepath = labels_filepath

    def generate_feature_vector(self, img_data):
        """
        Given a 2D matrix of image, generates the feature vector (x, y).
        x is the feature vector, y is the label
        """
        pass

    def get_image_matrix(self, filepath):
        """
        From a png filepath, get image matrix
        """
        return matplotlib.image.imread(filepath)

    def get_hit_rate_stats(self):
        """
        gets the threat probabilities in a useful form and returns
        a dataframe of the summary hit probabilities
        """
        infile = self.labels_filepath
        # pull the labels for a given patient
        df = pd.read_csv(infile)

        # Separate the zone and patient id into a df
        df['Subject'], df['Zone'] = df['Id'].str.split('_',1).str
        df = df[['Subject', 'Zone', 'Probability']]

        # make a df of the sums and counts by zone and calculate hit rate per zone, then sort high to low
        df_summary = df.groupby('Zone')['Probability'].agg(['sum','count'])
        df_summary['Zone'] = df_summary.index
        df_summary['pct'] = df_summary['sum'] / df_summary['count']
        df_summary.sort_values('pct', axis=0, ascending= False, inplace=True)
        
        return df_summary

    def chart_hit_rate_stats(self):
        """
        charts threat probabilities in desc order by zone
        """
        df_summary = self.df_summary

        fig, ax = plt.subplots(figsize=(15,5))
        sns.barplot(ax=ax, x=df_summary['Zone'], y=df_summary['pct']*100)

    def print_hit_rate_stats(self):
        """
        lists threat probabilities by zone.
        df_summary is a dataframe like that returned from self.get_hit_rate_stats(...)
        """
        df_summary = self.df_summary

        # print the table of values readbly
        print ('{:6s}   {:>4s}   {:6s}'.format('Zone', 'Hits', 'Pct %'))
        print ('------   ----- ----------')
        for zone in df_summary.iterrows():
            print ('{:6s}   {:>4d}   {:>6.3f}%'.format(zone[0], np.int16(zone[1]['sum']), zone[1]['pct']*100))
        print ('------   ----- ----------')
        print ('{:6s}   {:>4d}   {:6.3f}%'.format('Total ', np.int16(df_summary['sum'].sum(axis=0)), 
                                                 ( df_summary['sum'].sum(axis=0) / df_summary['count'].sum(axis=0))*100))

    def get_filepaths(self, directory):
        """
        retrieves a list of all filepaths from this directory
        """
        output = []

        onlyfiles = [
            f
            for f in os.listdir(os.path.realpath(directory))
            if os.path.isfile(os.path.join(os.path.realpath(directory), f))
        ]

        for k in onlyfiles:
            output.append(os.path.join(directory, k))

        return output

    def get_subject_labels(self, subject_id):
        """
        lists threat probabilities by zone and returns a df with the list of
        zones and contraband (0 or 1). subject id is the individual you want
        the threat zone labels for
        """
        infile = self.labels_filepath
        # read labels into a dataframe
        df = pd.read_csv(infile)

        # Separate the zone and subject id into a df
        df['Subject'], df['Zone'] = df['Id'].str.split('_',1).str
        df = df[['Subject', 'Zone', 'Probability']]
        threat_list = df.loc[df['Subject'] == subject_id]
        
        return threat_list

    def get_specific_threat_list(self, subject_id):
        df = pd.read_csv(self.labels_filepath)

        zone_list = df.loc[df['Id'].str.contains(subject_id)]
        zone_list_array = zone_list.values
        threat_list = []
        for item in zone_list_array:
            zone = int((item[0].split('_'))[1][4:])
            probability = item[1]
            threat_list.append([zone, probability])
        
        return sorted(threat_list, key=lambda x: x[0])

    def get_precise_threat_from_segment(self, subject_id, segment):
        #TODO: values hardcoded, move to global variables
        filepath = "../precise_labeling/xyfiles/threatcubes/"
        filename = subject_id + "_" + str(segment) + "_threatcube.txt"

        if not os.path.exists(filepath+filename):
            return -1

        file = open(filepath+filename)

        file_lines = file.readlines()

        # can identify x,y,z range using just 2 points in the cube
        point_1 = file_lines[0].split('\n')[0].split('\t')
        point_7 = file_lines[6].split('\n')[0].split('\t')

        # input ranges
        x_range = (int(point_1[0]),int(point_7[0]))
        y_range = (int(point_1[1]),int(point_7[1]))
        z_range = (int(point_1[2]),int(point_7[2]))

        # return range tuple
        return (x_range, y_range, z_range)

    def get_threatcubes(self, subject_id):
        threatcubes = []

        for segment in range (1,18):
            threatcubes.append(self.get_precise_threat_from_segment(subject_id, segment))

        return threatcubes

    def train(self):
        """
        Use SVM
        """
        pass


class TestingClassifier(SupervisedClassifier):
    """
    Classifier for regions 14 and 16
    """
    def __init__(self, labels_filepath):
        super(TestingClassifier, self).__init__(labels_filepath)

    def generate_feature_vector(self, img_path):
        """
        generates feature vectors from img_data (x, y)
        """
        subject_id = re.search(r"\/(\w+)_", img_path).group(1)
        img_data = matplotlib.image.imread(img_path)
        cropped_img = self.get_cropped_matrix(img_data)

        threats_list = self.get_specific_threat_list(subject_id)
        y = 0
        for zone, prob in threats_list:
            if zone == 14 or zone == 16:
                if prob == 1:
                    y = 1

        return (np.ndarray.flatten(cropped_img), y)

    def get_cropped_matrix(self, orig_img):
        """
        Crops to regions 14 and 16
        """
        return orig_img[500:660, 250:400]

    def get_feature_vectors(self, directory):
        """
        reads pngs from this directory and gets feature vector list
        """
        filelist = self.get_filepaths(directory)
        output = []
        for f in filelist:
            output.append(self.generate_feature_vector(f))
        return output

    def train(self, directory):
        """
        use svm
        """
        fvs = self.get_feature_vectors(directory)
        random.shuffle(fvs)
        num_fvs = len(fvs)
        ind_test = int(num_fvs / 10 * 8)
        ind = 0

        training_data = []
        training_labels = []
        test_data = []
        test_labels = []

        for feature, label in fvs:
            if ind < ind_test:
                training_data.append(feature)
                training_labels.append(label)
            else:
                test_data.append(feature)
                test_labels.append(label)
            ind += 1

        ### PCA ###
        print("--- PCA Output ---")

        print(len(test_data + training_data))
        pca = PCA(n_components=15)
        pca.fit(test_data + training_data)

        print(pca.components_.shape)
        print("Explained variance per component:")
        print(pca.explained_variance_ratio_)

        # plot first (x) and second (y) components
        plt.scatter(pca.components_[0], pca.components_[1])
        plt.show()

        ### SVM ###
        print("--- SVM Output ---")

        # create the classifier
        svm_classifier = svm.SVC(gamma=0.001)

        # we learn on the training data
        svm_classifier.fit(training_data, training_labels)
        print("Done training.")

        # now predict the threat
        predicted = svm_classifier.predict(test_data)

        # output results
        print("Classification report for classifier %s:\n%s\n"
              % (svm_classifier, metrics.classification_report(test_labels, predicted)))
        print("Confusion matrix:\n%s" % metrics.confusion_matrix(test_labels, predicted))

        ### RandomForestClassifier ###
        print("--- RandomForestClassifier Output ---")

        forest_classifier = RandomForestClassifier(n_estimators=10)

        #learn
        forest_classifier = forest_classifier.fit(training_data, training_labels)
        print("Done training.")

        # predict
        predicted = forest_classifier.predict(test_data)

        # output results
        print("Classification report for classifier %s:\n%s\n"
              % (forest_classifier, metrics.classification_report(test_labels, predicted)))
        print("Confusion matrix:\n%s" % metrics.confusion_matrix(test_labels, predicted))

# class DeepCNN(SupervisedClassifier):
#     """
#     This is an implementation of the deep
#     convolutional neural network to use for
#     a3d things
#     """
#     def __init__(self, labels_filepath):
#         """
#         pass
#         """
#         super(DeepCNN, self).__init__(labels_filepath)

#     def train(self, batch_in):
#         x = tf.placeholder(tf.float32, shape=[None, 512])
#         y_ = tf.placeholder(tf.float32, shape=[None, 2])

#         # First Convolutional Layer
#         W_conv1 = self.weight_variable([5, 5, 1, 32])
#         b_conv1 = self.bias_variable([32])

#         x_image = tf.reshape(x, [-1, 16, 16, 1])

#         h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1)
#         h_pool1 = self.max_pool_2x2(h_conv1)

#         # Second Convolutional Layer
#         W_conv2 = self.weight_variable([5, 5, 32, 64])
#         b_conv2 = self.bias_variable([64])

#         h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
#         h_pool2 = self.max_pool_2x2(h_conv2)

#         # Densely Connected Layer
#         W_fc1 = self.weight_variable([7 * 7 * 64, 1024])
#         b_fc1 = self.bias_variable([1024])

#         h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
#         h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#         # Dropout
#         keep_prob = tf.placeholder(tf.float32)
#         h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#         # Readout
#         W_fc2 = self.weight_variable([1024, 2])
#         b_fc2 = self.bias_variable([2])

#         y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

#         # TRAIN!
#         cross_entropy = tf.reduce_mean(
#             tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)
#         )
#         train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#         correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
#         accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#         with tf.Session() as sess:
#             sess.run(tf.global_variables_initializer())
#             i = 0
#             for batch in batch_in:
#                 if not batch.data.all():
#                     continue
#                 if batch.data.shape != (8, 8, 8):
#                     continue
#                 if i % 100 == 0:
#                     train_accuracy = accuracy.eval(feed_dict={
#                         x: batch.data, y_: batch.threat, keep_prob: 1.0
#                     })
#                     print('step %d, training accuracy %g' % (i, train_accuracy))
#                 train_step.run(feed_dict={x: batch.data, y_: batch.threat, keep_prob: 0.5})

#                 print('test accuracy %g' % accuracy.eval(feed_dict={
#                     x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
#                 i += 1

#     def weight_variable(self, shape):
#         initial = tf.truncated_normal(shape, stddev=0.1)
#         return tf.Variable(initial)

#     def bias_variable(self, shape):
#         initial = tf.constant(0.1, shape=shape)
#         return tf.Variable(initial)

#     def conv3d(self, x, W):
#         return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')

#     def conv2d(self, x, W):
#         return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

#     def max_pool_2x2x2(self, x):
#         return tf.nn.max_pool(x, ksize=[1, 2, 2, 2], strides=[1, 2, 2, 2], padding='SAME')

#     def max_pool_2x2(self, x):
#         return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


class SCPerceptron(SupervisedClassifier):
    """
    The most basic classifier you can think of. The perceptron!
    """
    pass


class SCSupportVectorMachine(SupervisedClassifier):
    """
    Make dem planes
    """
    pass


class WeakClassifiers(SupervisedClassifier):
    """
    Something
    """
    def symmetry_classifier(self):
        """
        The symmetry is off!
        """
        pass

    def classifier(self):
        """
        """
        pass


class SCAdaBoost(WeakClassifiers, SupervisedClassifier):
    """

    """
    pass
