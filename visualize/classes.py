from __future__ import print_function
from __future__ import division
from abc import ABCMeta
from matplotlib import pyplot as plt
import cv2
import numpy as np
import os
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import Queue

from constants import *


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

        return im2, sorted(contours), hierarchy

    def find_and_visualize_contours_for_slice(self, slice_number):
        # Get initial slice and visualize it in grayscale
        trans_data = self.img_data.transpose()

        # (https://stackoverflow.com/questions/30249053/python-opencv-drawing-errors-after-manipulating-array-with-numpy)
        trans_data_copy = trans_data.copy() #<-- look above for explanation
        gray_im = self.convert_to_grayscale(trans_data_copy[slice_number])
        plt.ion()
        fig = plt.figure()
        plt.imshow(gray_im)

        im2,contours,hierarchy = self.get_contours(trans_data_copy[slice_number])
        cv2.drawContours(gray_im, contours, -1, (0,255,0), 3)

        # Visualize the min enclosing circles around each of the contours
        # that our function latches onto
        for cnt in contours:
            (x,y),radius = cv2.minEnclosingCircle(cnt)
            center = (int(x),int(y))
            radius = int(radius)
            plt.imshow(cv2.circle(gray_im,center,radius,(0,255,0),2))

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

        region_matrices = {};
        img = self.img_data
        img_data_trans = img.transpose()

        # iterate through each segment zone
        for i in range(17):
            region_matrices[i] = self.crop(img_data_trans, sector_crop_list[i])

        return region_matrices
        ex
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
        self.df_summary = self.get_hit_rate_stats()

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
