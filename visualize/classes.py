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

from constants import *


class BodyScan(object):
    """
    Main class for body scan data.
    """
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
