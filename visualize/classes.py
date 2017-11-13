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
import pandas as pd
import scipy.stats as stats
import png
import re
import math
import random
import seaborn as sns
import nibabel as nib
import errno
from sklearn import svm, metrics, tree, naive_bayes
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.decomposition import PCA
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
        self.person_id = re.search(r"(\w+)\.", filepath).group(1)
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
        dirname = os.path.dirname(os.path.realpath(file_path))
        fpath = os.path.join(dirname, self.person_id)
        fname = fpath + "_projection.png"
        matrix = self.flatten_max(axis=axis)
        self.write_slice_to_img(matrix, fname)
        return(fname)

    def write_slice_to_img(self, slic, filename):
        """
        Given a slice, writes it to a png
        """
        slc = (slic - np.average(slic)) / np.std(slic)
        slc = slc - np.min(slc)
        slic = slc / np.max(slc) * 255
        if not filename.endswith(".png"):
            filename += ".png"
        with open(filename, "wb") as f:
            w = png.Writer(len(slic[0]), len(slic), greyscale=True)
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

    def write_slice_to_img(self, slic, filename):
        """
        Given a slice, writes it to a png
        """
        directory = os.path.dirname(os.path.realpath(filename))
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        slc = (slic - np.average(slic)) / np.std(slic)
        slc = slc - np.min(slc)
        slic = slc / np.max(slc) * 255
        if not filename.endswith(".png"):
            filename += ".png"
        with open(filename, "wb") as f:
            w = png.Writer(len(slic[0]), len(slic), greyscale=True)
            w.write(f, slic)

    def normalize(self, image):
        """
        takes an image and normalizes it (0 mean, 1 variance)
        """
        return (image - np.average(image)) / np.std(image)

    def train(self):
        """
        Use SVM
        """
        pass


class FeatureGenerator(SupervisedClassifier):
    def generate_negative_features(
            self,
            directory,
            minsize=24,
            maxsize=50,
            num=25,
            var_thresh_min=0.0692,
            var_thresh_max=4.129,
            mean_thresh_min=-.238,
            mean_thresh_max=3.425,
            dest="../data/negative_examples/"):
        """
        Generates positive features (no threat). num is num randomly generated
        features per file.
        """
        paths = self.get_filepaths(directory)
        for path in paths:
            if not path.endswith(".a3d"):
                continue
            print("Generating {}".format(path))
            person_id = re.search(r"(\w+)\.", path).group(1)
            threat_list = self.get_specific_threat_list(person_id)

            is_clear = True
            for m, k in threat_list:
                if k == 1:
                    is_clear = False
                    break

            if not is_clear:
                continue

            bs = BodyScan(path)
            flattened = bs.flatten_max()
            flattened = self.normalize(flattened)
            n = len(flattened)
            m = len(flattened[0])

            ct = 0
            while ct < num:
                size = random.randint(minsize, maxsize)
                ds = int(size/2)
                x = random.randint(ds, n - 1 - ds)
                y = random.randint(ds, m - 1 - ds)

                slic = flattened[x-ds:x+ds, y-ds:y+ds]
                var = np.var(slic)
                mean = np.mean(slic)

                # make sure generated data looks like the threats
                if var < var_thresh_min:
                    continue
                if var > var_thresh_max:
                    continue
                if mean < mean_thresh_min:
                    continue
                if mean > mean_thresh_max:
                    continue

                prob = stats.norm.pdf(mean, 1.3043641, .6739)
                rnum = np.random.rand()

                if rnum > prob:
                    continue

                outfile = os.path.join(dest, "{}_{}".format(
                    bs.person_id,
                    ct
                ))
                bs.write_slice_to_img(slic, outfile)
                ct += 1

    def squarize(self, x1, x2, y1, y2, xlim, ylim, make_bigger=True):
        """
        Squarizes the bounds
        """
        # adjust it to a square
        xdist = abs(x2 - x1)
        ydist = abs(y2 - y1)

        if xdist > ydist:
            diff = xdist - ydist
            while diff % 2 == 1:
                y1 -= 1
                diff = xdist - abs(y2 - y1)

            if make_bigger:
                y2 += int(diff / 2)
                y1 -= int(diff / 2)
            else:
                x2 -= int(diff / 2)
                x1 += int(diff / 2)

            if y2 > ylim:
                d = y2 - ylim
                y2 -= d
                y1 -= d

            if y1 < 0:
                d = 0 - y1
                y2 += d
                y1 += d

        elif ydist > xdist:
            diff = ydist - xdist
            while diff % 2 == 1:
                x1 -= 1
                diff = ydist - abs(x2 - x1)

            if make_bigger:
                x2 += int(diff / 2)
                x1 -= int(diff / 2)
            else:
                y2 -= int(diff / 2)
                y1 += int(diff / 2)

            if x2 > xlim:
                d = x2 - xlim
                x2 -= d
                x1 -= d

            if x1 < 0:
                d = 0 - x1
                x2 += d
                x1 += d

        return (x1, x2, y1, y2)

    def get_coords_from_file(self, coords_file):
        """
        gets the coordindates from threatcube file
        """
        x1 = y1 = z1 = x2 = y2 = z2 = None
        with open(coords_file, "r") as file:
            for row in file:
                coords = row.strip().split()
                if not x1:
                    x1 = int(coords[1])
                if not y1:
                    y1 = int(coords[0])
                if not z1:
                    z1 = int(coords[2])

                if x1 >= int(coords[1]):
                    x1 = int(coords[1])
                else:
                    x2 = int(coords[1])

                if y1 >= int(coords[0]):
                    y1 = int(coords[0])
                else:
                    y2 = int(coords[0])

                if z1 >= int(coords[2]):
                    z1 = int(coords[2])
                else:
                    z2 = int(coords[2])
        return (x1, x2, y1, y2, z1, z2)

    def check_neighbors(self, seg_img, x, y):
        """
        returns True if neighbors are all same
        returns False if neighbors aren't all same
        """
        val = seg_img[x][y]
        if x > 0:
            x1 = seg_img[x-1][y]
            if x1 != val:
                return False
        if x < len(seg_img) - 1:
            x1 = seg_img[x+1][y]
            if x1 != val:
                return False
        if y > 0:
            x1 = seg_img[x][y-1]
            if x1 != val:
                return False
        if y < len(seg_img[0]) - 1:
            x1 = seg_img[x][y+1]
            if x1 != val:
                return False

        return True

    def get_img_from_nii(self, nii_path):
        """
        imports the mask from nii path and processes it.
        """
        # load it
        img = nib.load(nii_path)

        # orient it correctly
        seg_img = np.rot90(np.rot90(img.get_data()))

        # make it into numbers
        seg_img = seg_img / np.min(seg_img)

        # fill it with integers
        img = np.zeros(seg_img.shape, dtype=int)
        for i in range(len(seg_img)):
            for j in range(len(seg_img[0])):
                img[i][j] = int(round(seg_img[i][j]))

        return img

    def compare_ys(self, box1, box2):
        """
        returns True if box1 is on the left
        returns False if box2 is on the left
        """
        b1av = (box1[2] + box1[3]) / 2
        b2av = (box2[2] + box2[3]) / 2

        if b1av < b2av:
            return True
        else:
            return False

    def clean_up_regions(self, boxes):
        """
        Takes the boxes and assigns regions to them
        """
        output = {}
        # 2, 4
        res = self.compare_ys(boxes[3], boxes[4])
        if res:
            output[2] = boxes[3]
            output[4] = boxes[4]
        else:
            output[2] = boxes[4]
            output[4] = boxes[3]

        # 1, 3
        res = self.compare_ys(boxes[6], boxes[7])
        if res:
            output[1] = boxes[6]
            output[3] = boxes[7]
        else:
            output[1] = boxes[7]
            output[3] = boxes[6]

        # 5
        output[5] = boxes[8]

        # 6, 7
        res = self.compare_ys(boxes[9], boxes[10])
        if res:
            output[6] = boxes[9]
            output[7] = boxes[10]
        else:
            output[6] = boxes[10]
            output[7] = boxes[9]

        res1 = self.compare_ys(boxes[11], boxes[12])
        res2 = self.compare_ys(boxes[12], boxes[13])
        res3 = self.compare_ys(boxes[11], boxes[13])

        if res1 and res2 and res3:
            output[8] = boxes[11]
            output[9] = boxes[12]
            output[10] = boxes[13]
        elif res1 and not res2 and res3:
            output[8] = boxes[11]
            output[9] = boxes[13]
            output[10] = boxes[12]
        elif not res1 and res2 and res3:
            output[8] = boxes[12]
            output[9] = boxes[11]
            output[10] = boxes[13]
        elif not res1 and res2 and not res3:
            output[8] = boxes[12]
            output[9] = boxes[13]
            output[10] = boxes[11]
        elif res1 and not res2 and not res3:
            output[8] = boxes[13]
            output[9] = boxes[11]
            output[10] = boxes[12]
        else:
            output[8] = boxes[13]
            output[9] = boxes[12]
            output[10] = boxes[11]

        # 11, 12
        res = self.compare_ys(boxes[14], boxes[15])
        if res:
            output[11] = boxes[14]
            output[12] = boxes[15]
        else:
            output[11] = boxes[15]
            output[12] = boxes[14]

        # 13, 14
        res = self.compare_ys(boxes[16], boxes[17])
        if res:
            output[13] = boxes[16]
            output[14] = boxes[17]
        else:
            output[13] = boxes[17]
            output[14] = boxes[16]

        # 15, 16
        res = self.compare_ys(boxes[18], boxes[19])
        if res:
            output[15] = boxes[18]
            output[16] = boxes[19]
        else:
            output[15] = boxes[19]
            output[16] = boxes[18]

        return output

    def get_bounding_boxes(self, seg_img):
        """
        given a seg_img, returns the bounding boxes for
        each of the segmented region.
        1: right hand
        2: left hand
        3: 2
        4: 4
        5: head
        6: 1
        7: 3
        8: 5 / 17
        9: 7
        10: 6
        11: 10
        12: 8
        13: 13
        14: 12
        15: 11
        16: 14
        17: 13
        18: 15
        19: 16
        20: background
        """
        ziyi_table = {
            3: 2,
            4: 4,
            6: 1,
            7: 3,
            8: 5,
            9: 7,
            10: 6,
            11: 10,
            12: 8,
            13: 13,
            14: 12,
            15: 11,
            16: 14,
            17: 13,
            18: 15,
            19: 16
        }

        output = {}

        for i in range(len(seg_img)):
            for j in range(len(seg_img[0])):
                val = seg_img[i][j]
                if not self.check_neighbors(seg_img, i, j):
                    val = 20
                if val not in output:
                    output[val] = [2151234, -124512, 12351234, -124612345]
                if i < output[val][0]:
                    output[val][0] = i
                if i > output[val][1]:
                    output[val][1] = i + 1
                if j < output[val][2]:
                    output[val][2] = j
                if j > output[val][3]:
                    output[val][3] = j + 1

        return self.clean_up_regions(output)

    def generate_positive_features(self, precise_labels_dir, dest="../data/positive_examples", make_bigger=True):
        """
        from precise labels, generates positive square features.
        """
        files = self.get_filepaths(precise_labels_dir)
        for f in files:
            if f.endswith("threatcube.txt"):
                (x1, x2, y1, y2, z1, z2) = self.get_coords_from_file(f)
            else:
                continue
            (xy_x1, xy_x2, xy_y1, xy_y2) = self.squarize(x1, x2, y1, y2, 660, 512, make_bigger=make_bigger)
            (xz_x1, xz_x2, xz_z1, xz_z2) = self.squarize(x1, x2, z1, z2, 660, 512, make_bigger=make_bigger)
            (yz_y1, yz_y2, yz_z1, yz_z2) = self.squarize(y1, y2, z1, z2, 512, 512, make_bigger=make_bigger)

            # now generate file!
            pid = re.search(r"(\w+)_\d+_threat", f).group(1)
            reg = re.search(r"(\d+)_threat", f).group(1)
            fname = pid + "_" + reg
            img = self.get_image_matrix(os.path.join("D:/590Data", pid) + "_projection.png")
            threat = img[xy_x1:xy_x2, xy_y1:xy_y2]

            self.write_slice_to_img(threat, os.path.join(dest, fname))

    def generate_features_from_seg(
        self,
        seg_dir="../data/2D_segmentation",
        dest="../data",
        data_dir="D:/590Data"
            ):
        """
        Given the 2D segmentation, generates positive examples from it.
        It is positive if it's actually a body.
        Negative if not body part.
        """
        files = [f for f in self.get_filepaths(seg_dir) if f.endswith(".nii")]
        for f in files:
            print("Now generating from {}".format(f))
            pid = re.search(r"(\w+)_label\.nii", f).group(1)
            th_list = self.get_specific_threat_list(pid)

            threat_list = {k: v for k, v in th_list}
            mask = self.get_img_from_nii(f)

            impath = os.path.join(data_dir, pid + "_projection.png")
            img = self.get_image_matrix(impath)
            try:
                boxes = self.get_bounding_boxes(mask)
            except:
                continue
            for reg, box in boxes.items():
                is_threat = False
                try:
                    if threat_list[reg] == 1:
                        is_threat = True
                except KeyError:
                    continue
                if reg == 5 and threat_list[17] == 1:
                    is_threat = True

                if is_threat:
                    subfolder = "negative_examples"
                else:
                    subfolder = "positive_examples"

                subsub = str(reg)

                sq = self.squarize(box[0], box[1], box[2], box[3], 660, 512)

                example = img[sq[0]:sq[1], sq[2]:sq[3]]

                outdir = os.path.join(os.path.realpath(dest), subfolder)
                outdir = os.path.join(outdir, subsub)
                outdir = os.path.join(outdir, pid + "_" + str(reg) + ".png")

                self.write_slice_to_img(example, outdir)


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
        subject_id = re.search(r"(\w+)_", img_path).group(1)
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


class SCAdaBoost(SupervisedClassifier):
    """
    An Implementation of Viola and Jones.
    """
    def load_examples(self, pos_dir, neg_dir, pn=0, nn=0):
        """
        Loads examples from these directories.
        in integral sum format.
        number of positive examples pn (choose randomly)
        number of negative examples nn (choose randomly)
        """
        print("loading examples")
        x = []
        p_files = self.get_filepaths(pos_dir)
        n_files = self.get_filepaths(neg_dir)

        if pn == 0:
            pn = len(p_files)
        if nn == 0:
            nn = len(n_files)

        random.shuffle(p_files)
        random.shuffle(n_files)

        with open("positive_examples.txt", "w") as f:
            for k in range(pn):
                img = self.get_image_matrix(p_files[k])
                img = self.calculate_integral_sum(img)
                x.append((img, 1))
                f.write(os.path.realpath(p_files[k]) + "\n")
        with open("negative_examples.txt", "w") as f:
            for k in range(nn):
                img = self.get_image_matrix(n_files[k])
                img = self.calculate_integral_sum(img)
                x.append((img, -1))
                f.write(os.path.realpath(n_files[k]) + "\n")

        self.x = x

    def generate_features(self, img_path):
        """
        given a square image with 0 mean and 1 variance, (use 24)
        returns a set of features with haar-like feature vector.
        returns (sum, type, i, j, w, h)
        """
        # Preprocess the image
        img = self.get_image_matrix(img_path)
        i_img = self.calculate_integral_sum(img)

        features = []

        print("Generating feature vectors")
        n = len(img)  # viola and jones uses 24 pixels here.
        for i in range(n):
            for j in range(n):
                for w in range(1, n + 1):
                    for h in range(1, n + 1):
                        if i + h < n and j + 2 * w < n:
                            S = self.sum_t1(i_img, i, j, w, h)
                            features.append((S, 1, i, j, w, h))

                        if i + h < n and j + 3 * w < n:
                            S = self.sum_t2(i_img, i, j, w, h)
                            features.append((S, 2, i, j, w, h))

                        if i + 2 * h < n and j + w < n:
                            S = self.sum_t3(i_img, i, j, w, h)
                            features.append((S, 3, i, j, w, h))

                        if i + 3 * h < n and j + w < n:
                            S = self.sum_t4(i_img, i, j, w, h)
                            features.append((S, 4, i, j, w, h))

                        if i + 2 * h < n and j + 2 * w < n:
                            S = self.sum_t5(i_img, i, j, w, h)
                            features.append((S, 5, i, j, w, h))

        self.features = features
        return features

    def get_feature_vals_for_all(self, n=24):
        """
        herpaderp.
        feature_vecs is a 2 dim array with [feature][x_i]
        where x_i is the (val, label) lies.
        """
        print("Calculating feature vectors for all examples")
        self.examples = []

        ct = 0
        for x in self.x:
            ct += 1
            fv = []
            for f in self.features:
                fv.append(self.scale_feature(x[0], f, n=n))
            self.examples.append([tuple(fv), x[1], 0])
            print("{} / {} done".format(ct, len(self.x)))

    def __init__(self, labels_filepath, pos_dir, neg_dir, example_file):
        """
        Initializes training example matrix

        initializes the following

        self.features: an array of features
        self.examples: an array of [(feature values), label, weights]
        (weights to be initialized later.)
        """
        super(SCAdaBoost, self).__init__(labels_filepath)
        self.generate_features(example_file)

        # These should be run if you want to train.
        # self.load_examples(pos_dir, neg_dir)
        # self.get_feature_vals_for_all()
        # del self.x

    def calculate_integral_sum(self, img):
        """
        calculates the integral sum of the image.
        """
        img = self.normalize(img)
        output = np.zeros((len(img), len(img[0])))
        output[0][0] = img[0][0]
        for i in range(len(img)):
            for j in range(len(img[0])):
                output[i][j] = img[i][j]
                if (j - 1 >= 0):
                    output[i][j] += output[i][j - 1]
                if (i - 1 >= 0):
                    output[i][j] += output[i - 1][j]
                if (i - 1 >= 0) and (j - 1 >= 0):
                    output[i][j] -= output[i - 1][j - 1]

        return output

    def get_sum(self, img, x1, x2, y1, y2):
        """
        given (x1, y1) and (x2, y2) and integral image,
        returns the sum within that square.
        """
        n = len(img)
        if x1 < 0:
            x1 = 0
        if x2 >= n:
            x2 = n - 1
        if y1 < 0:
            y1 = 0
        if y2 >= n:
            y2 = 0
        S = img[x1][y1]
        S += img[x2][y2]
        S -= img[x1][y2]
        S -= img[x2][y1]

        return S

    def sum_t1(self, i_img, i, j, w, h):
        """
        gets the sum of type1 feature.
        """
        S1 = self.get_sum(i_img, i, i + h, j, j + w)
        S2 = self.get_sum(i_img, i, i + h, j + w, j + 2 * w)

        return S1 - S2

    def sum_t2(self, i_img, i, j, w, h):
        S1 = self.get_sum(i_img, i, i + h, j, j + w)
        S2 = self.get_sum(i_img, i, i + h, j + w, j + 2 * w)
        S3 = self.get_sum(i_img, i, i + h, j + 2 * w, j + 3 * w)

        return S1 - S2 + S3

    def sum_t3(self, i_img, i, j, w, h):
        S1 = self.get_sum(i_img, i, i + h, j, j + w)
        S2 = self.get_sum(i_img, i + h, i + 2 * h, j, j + w)

        return S1 - S2

    def sum_t4(self, i_img, i, j, w, h):
        S1 = self.get_sum(i_img, i, i + h, j, j + w)
        S2 = self.get_sum(i_img, i + h, i + 2 * h, j, j + w)
        S3 = self.get_sum(i_img, i + 2 * h, i + 3 * h, j, j + w)

        return S1 - S2 + S3

    def sum_t5(self, i_img, i, j, w, h):
        S1 = self.get_sum(i_img, i, i + h, j, j + w)
        S2 = self.get_sum(i_img, i + h, i + 2 * h, j, j + w)
        S3 = self.get_sum(i_img, i, i + h, j + w, j + 2 * w)
        S4 = self.get_sum(i_img, i + h, i + 2 * h, j + w, j + 2 * w)

        return S1 - S2 - S3 + S4

    def scale_feature(self, i_img, f, n=24):
        """
        given a square image and n, it scales the feature and yeah.
        returns it with scaled sum values.
        i_img is normalized integral sum image
        f is the (sum, type, i, j, w, h) feature tuple.
        """
        e = len(i_img)

        s, t, i, j, w, h = f

        i = round(i * e / n)
        j = round(j * e / n)

        if t == 1:
            a = 2 * w * h
            h = round(h * e / n)
            w = min(int(round(1 + 2 * w * e / n) / 2), 2 * (e - j + 1))

            return self.sum_t1(i_img, i, j, w, h) * a / (2 * w * h)
        elif t == 2:
            a = 3 * w * h
            h = round(h * e / n)
            w = min(int(round(1 + 3 * w * e / n) / 3), 3 * (e - j + 1))

            return self.sum_t2(i_img, i, j, w, h) * a / (3 * w * h)
        elif t == 3:
            a = 2 * w * h
            w = round(w * e / n)
            h = min(int(round(1 + 2 * h * e / n) / 2), 2 * (e - i + 1))

            return self.sum_t3(i_img, i, j, w, h) * a / (2 * w * h)
        elif t == 4:
            a = 3 * w * h
            w = round(w * e / n)
            h = min(int(round(1 + 3 * h * e / n) / 3), 3 * (e - i + 1))

            return self.sum_t4(i_img, i, j, w, h) * a / (3 * w * h)
        elif t == 5:
            a = 4 * w * h
            w = min(int(round(1 + 2 * w * e / n) / 2), 2 * (e - j + 1))
            h = min(int(round(1 + 2 * h * e / n) / 2), 2 * (e - i + 1))

            return self.sum_t5(i_img, i, j, w, h) * a / (4 * w * h)
        else:
            print("FEATURE TYPE NOT FOUND")

        return None

    def decision_stump(self, examples, f_i):
        """
        examples are (pi_f * x_i, y, w) sorted, which is the feature
        value of feature f of ith example and y is the label.
        w is example weights

        This is a weird function
        """
        tau = examples[0][0][f_i] - 1
        M = 0
        E = 2

        Wpp = 0  # W^+_1
        Wpn = 0  # Positive examples lower than thresh
        Wnp = 0  # Negative examples bigger than thresh
        Wnn = 0  # Negative examples lower than thresh

        for k in range(len(examples)):
            fv, y, w = examples[k]
            fv = fv[f_i]
            if fv >= tau and y == 1:
                Wpp += w
            elif fv >= tau and y == -1:
                Wpn += w
            elif fv < tau and y == 1:
                Wnp += w
            elif fv < tau and y == -1:
                Wnn += w
            else:
                print("WOW THAT DIDN'T WORK")
                print(examples[k])

        Wnp = 0
        Wnn = 0

        j = -1
        tau_hat = tau
        M_hat = M
        n = len(examples)

        while True:
            ep = Wnp + Wpn
            en = Wpp + Wnn

            if ep < en:
                E_hat = ep
                T_hat = 1
            else:
                E_hat = en
                T_hat = -1

            if E_hat < E or E == E_hat and M_hat > M:
                E = E_hat
                tau = tau_hat
                M = M_hat
                T = T_hat

            if j == n - 1:
                break

            j += 1
            while True:
                fv, y, w = examples[j]
                fv = fv[f_i]
                if y == -1:
                    Wnn += w
                    Wpn -= w
                else:
                    Wnp += w
                    Wpp -= w
                if j == n - 1 or examples[j + 1][0][f_i] != fv:
                    break
                else:
                    fv2 = examples[j + 1][0][f_i]
                    j += 1
            if j == n - 1:
                tau_hat = examples[n-1][0][f_i] + 1
                M_hat = 0
            else:
                fv2 = examples[j + 1][0][f_i]
                tau_hat = (fv + fv2) / 2
                M_hat = fv2 - fv

        return (tau, T, E, M)

    def best_stump(self):
        """
        w is just the weights
        features is an array of all the features
        returns (tau, T, E, M), feature
        """
        best_error = 2
        best_stump = (0, 0, 0, -9999999)
        best_feature_i = 0
        for k in range(len(self.features)):
            # get kth feature
            self.examples.sort(key=lambda x: x[0][k])
            stump = self.decision_stump(self.examples, k)

            E = stump[2]
            M = stump[3]

            if E < best_error or best_error == E and M > best_stump[3]:
                best_error = E
                best_stump = stump
                best_feature_i = k

        return best_stump, best_feature_i

    def eval_stump(self, stump, val):
        """
        evaluates h(x) using the given decision stump and value.
        """
        (tau, T, E, M) = stump
        if T == 1:
            if val >= tau:
                return 1
            else:
                return -1
        else:
            if val < tau:
                return 1
            else:
                return -1

    def adaboost(self, T, save_to_file=False):
        """
        x is a vector of training examples in self.examples
        DO THE ADABOOST!
        """
        # Initialize weights
        for k in range(len(self.examples)):
            self.examples[k][2] = 1 / len(self.examples)

        alphas = []
        stumps = []

        for t in range(T):
            stump, f_i = self.best_stump()

            E = 0
            for x in self.examples:
                f_val = x[0][f_i]
                dec = self.eval_stump(stump, f_val)
                if dec != x[1]:
                    E += x[2]

            if E == 0:
                E = 1e-16
            if E <= 1e-16 and t == 0:
                return stump, f_i
            else:
                alph = (0.5) * math.log((1 - E) / E)
                print("{}\t{}\t{}\t{}\t{}".format(E, stump[3], stump[1], f_i, alph))
                alphas.append(alph)
                stumps.append((stump, f_i))
                if save_to_file:
                    with open("ada_output.txt", "a") as fil:
                        fil.write("{}|{}|{}\n".format(str(stump), str(f_i), str(alph)))
                for k in range(len(self.examples)):
                    w = self.examples[k][2]
                    new_w = 0

                    yi = self.examples[k][1]
                    f_val = self.examples[k][0][f_i]
                    dec = self.eval_stump(stump, f_val)

                    if dec != yi:
                        new_w = w / 2 * 1 / E
                    else:
                        new_w = w / 2 * 1 / (1 - E)

                    self.examples[k][2] = new_w

        return stumps, alphas

    def show_feature(self, f, n=24):
        """
        shows feature.
        """
        s, t, x, y, w, h = f
        img = np.zeros((n, n))
        if t == 1:
            for n in range(x, x + h):
                for m in range(y, y + w):
                    img[n][m] = 1
            for n in range(x, x + h):
                for m in range(y + w, y + 2 * w):
                    img[n][m] = -1
        if t == 2:
            for n in range(x, x + h):
                for m in range(y, y + w):
                    img[n][m] = 1
            for n in range(x, x + h):
                for m in range(y + w, y + 2 * w):
                    img[n][m] = -1
            for n in range(x, x + h):
                for m in range(y + 2 * w, y + 3 * w):
                    img[n][m] = 1
        if t == 3:
            for n in range(x, x + h):
                for m in range(y, y + w):
                    img[n][m] = 1
            for n in range(x + h, x + 2 * h):
                for m in range(y, y + w):
                    img[n][m] = -1
        if t == 4:
            for n in range(x, x + h):
                for m in range(y, y + w):
                    img[n][m] = 1
            for n in range(x + h, x + 2 * h):
                for m in range(y, y + w):
                    img[n][m] = -1
            for n in range(x + 2 * h, x + 3 * h):
                for m in range(y, y + w):
                    img[n][m] = 1
        if t == 5:
            for n in range(x, x + h):
                for m in range(y, y + w):
                    img[n][m] = 1
            for n in range(x + h, x + 2 * h):
                for m in range(y, y + w):
                    img[n][m] = -1
            for n in range(x, x + h):
                for m in range(y + w, y + 2 * w):
                    img[n][m] = -1
            for n in range(x + h, x + 2 * h):
                for m in range(y + w, y + 2 * w):
                    img[n][m] = 1

        plt.imshow(img)
        plt.show()

    def show_feature_num(self, f_i):
        """
        shows the ith index's feature.
        """
        self.show_feature(self.features[f_i])

    def classify(self, img, classifiers, thresh=0.125):
        """
        uses the array of weak classifiers to classify.
        [(stump, f_i, alpha)]
        """
        i_img = self.calculate_integral_sum(img)
        output = 0
        for stump, f_i, alpha in classifiers:
            fv = self.scale_feature(i_img, self.features[f_i])
            output += self.eval_stump(stump, fv) * alpha

        return np.sign(output)

    def get_classifiers(self, filepath):
        file = filepath
        classifiers = []

        with open(file, "r") as f:
            for line in f:
                elems = line.strip().split("|")
                stump_strs = re.sub(r"\s+", "", elems[0].strip("(").strip(")")).split(",")
                stump = (float(stump_strs[0]), int(stump_strs[1]), float(stump_strs[2]), float(stump_strs[3]))
                f_i = int(elems[1])
                alpha = float(elems[2])

                classifiers.append((stump, f_i, alpha))

        return classifiers

    def test(self, pos_dir, neg_dir, ada_output="ada_output.txt"):
        """
        tests the labels using the algorithm.
        """
        file = ada_output

        self.classifiers = self.get_classifiers(file)
        classifiers = self.classifiers

        fps_p = self.get_filepaths(pos_dir)
        fps_n = self.get_filepaths(neg_dir)

        p_train = []
        n_train = []

        with open("positive_examples.txt", "r") as f:
            for s in f:
                pid = re.search(r"(\w+)_\d+\.png", s).group(0)
                p_train.append(pid)
        with open("negative_examples.txt", "r") as f:
            for s in f:
                pid = re.search(r"(\w+)_\d+\.png", s).group(0)
                n_train.append(pid)

        tn = 0
        tp = 0
        fp = 0
        fn = 0
        ap = 0
        an = 0

        for fil in fps_p:
            pid = re.search(r"(\w+)_\d+\.png", fil).group(0)
            if pid in p_train:
                continue
            img = self.get_image_matrix(fil)
            res = self.classify(img, classifiers)
            ap += 1
            if res == 1:
                tp += 1
            else:
                fn += 1

        for fil in fps_n:
            pid = re.search(r"(\w+)_\d+\.png", fil).group(0)
            if pid in n_train:
                continue
            img = self.get_image_matrix(fil)
            res = self.classify(img, classifiers)
            if res == -1:
                tn += 1
            else:
                fp += 1
            an += 1

        print("Number of weak classifiers: {}".format(len(classifiers)))
        print("Confusion Matrix:\n{}\t{}\n{}\t{}".format(tn, fp, fn, tp))
        print("Accuracy: {}".format((tp + tn) / (ap + an)))
        print("Misclassification: {}".format((fp + fn) / (ap + an)))
        print("True Positive (Recall): {}".format(tp / ap))
        print("False Positive: {}".format(fp / an))
        print("Specificity: {}".format(tn / an))
        print("Precision: {}".format(tp / (tp + fp)))
        print("Prevalence: {}".format(ap / (ap + an)))


class OneClassSVM(SupervisedClassifier):
    def load_features(self, directory, num=None):
        filepaths = self.get_filepaths(directory)
        features = []
        ct = 0
        for fp in filepaths:
            if num:
                if ct > num:
                    break
                ct += 1
            img = self.get_image_matrix(fp)
            feat = np.ndarray.flatten(img)
            features.append(feat)

        print("positive vibes")
        self.clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
        self.clf.fit(features)

    def predict(self, filepath, size=100):
        """
        Scans the file and draws bounding box for threat.
        """
        ds = int(size/2)
        flattened = self.get_image_matrix(filepath)
        n = len(flattened)
        m = len(flattened[0])

        output = []

        for x in range(ds, n - 1 - ds, ds):
            for y in range(ds, m - 1 - ds, ds):

                slic = flattened[x-ds:x+ds, y-ds:y+ds]
                oned = np.ndarray.flatten(slic)
                p = self.clf.predict([oned])
                output.append((x, y, p))

        return output

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
