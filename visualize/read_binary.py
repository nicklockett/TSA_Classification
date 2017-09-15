# coding: utf-8

# Functions to read and view the competition files. 
# Most of the information in the file header is identical across scans and thus not relevant to the competition, except for
# data_scale_factor
#
# Header file is first 512 bytes of binary file
# Binary file is one of four formats
# .ahi
# .a3d
# .aps
# .a3daps


import numpy as np
import os
from os.path import splitext, basename, join, dirname, isdir
import matplotlib
# matplotlib.rc('animation', html='html5')
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation


im_fpath = "fdb996a779e5d65d043eaa160ec2f09f.aps"
# im_fpath = "/data2/seg/data/a3d/0043db5e8c819bffc15261b1f1ac5e42.a3d"

fname_id = splitext(basename(im_fpath))[0]
ext_no_dot = splitext(basename(im_fpath))[1][1:]
ani_file_out = join("output/", fname_id + "_" + ext_no_dot + ".mp4")


def read_header(infile):
	"""Read image header (first 512 bytes)
	"""
	h = dict()
	fid = open(infile, 'r+b')
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


def read_data(infile):
	"""Read any of the 4 types of image files (.ahi, .a3d, .aps, .a3daps), returns a numpy array of the image contents
	"""
	# Get file extension
	extension = splitext(infile)[1]

	# Get header
	h = read_header(infile)
	nx = int(h['num_x_pts'])
	ny = int(h['num_y_pts'])
	nt = int(h['num_t_pts'])

	# Get the data and scale it accordingly
	fid = open(infile, 'rb')
	fid.seek(512) # skip header

	if extension == '.aps' or extension == '.a3daps':
		if(h['word_type'] == 7): # float32
			data = np.fromfile(fid, dtype = np.float32, count = nx * ny * nt)
		elif(h['word_type'] == 4): # uint16
			data = np.fromfile(fid, dtype = np.uint16, count = nx * ny * nt)
		data = data * h['data_scale_factor'] # scaling factor
		data = data.reshape(nx, ny, nt, order = 'F').copy() # make N-d image
	elif extension == '.a3d':
		if(h['word_type'] == 7): # float32
			data = np.fromfile(fid, dtype = np.float32, count = nx * ny * nt)
		elif(h['word_type'] == 4): # uint16
			data = np.fromfile(fid, dtype = np.uint16, count = nx * ny * nt)
		data = data * h['data_scale_factor'] # scaling factor
		data = data.reshape(nx, nt, ny, order = 'F').copy() # make N-d image
	elif extension == '.ahi':
		data = np.fromfile(fid, dtype = np.float32, count = 2 * nx * ny * nt)
		data = data.reshape(2, ny, nx, nt, order = 'F').copy()
		real = data[0, :, :, :].copy()
		imag = data[1, :, :, :].copy()

	fid.close()

	if extension != '.ahi':
		return data
	else:
		return real, imag


def plot_image_aps_or_a3daps(path):
	"""
	# Example plotting function: .aps or .a3daps file animation
	"""
	data = read_data(path)
	fig = plt.figure(figsize = (16, 16))
	ax = fig.add_subplot(1, 1, 1)

	def animate(i):
		im = ax.imshow(np.flipud(data[:, :, i].transpose()), cmap = 'viridis')
		return [im]

	return animation.FuncAnimation(fig, animate, frames = range(0, data.shape[2]), interval = 200, blit = True)


def plot_image_a3d_vertical_slices(path):
	"""
	# Example plotting function: .a3d file animation. Slices along y-axis (y-axis is normal to slicing planes)
	"""
	data = read_data(path) # (x, z, y)
	fig = plt.figure(figsize = (16, 16))
	ax = fig.add_subplot(1, 1, 1)

	def animate(i):
		im = ax.imshow(data[:, :, i].transpose(), cmap = 'viridis')
		return [im]

	return animation.FuncAnimation(fig, animate, frames = range(0, data.shape[2]), interval = 200, blit = True)


def plot_image_a3d_horizontal_slices(path):
	"""
	# Example plotting function: .a3d file animation. Slices along x-axis (x-axis is normal to slicing planes)
	"""
	data = read_data(path) # (x, z, y)
	fig = plt.figure(figsize = (16, 16))
	ax = fig.add_subplot(1, 1, 1)

	def animate(i):
		im = ax.imshow(data[i, :, :].transpose(), cmap = 'viridis')
		return [im]

	return animation.FuncAnimation(fig, animate, frames = range(0, data.shape[0]), interval = 200, blit = True)


def plot_image_a3d_depth_slices(path):
	"""
	# Example plotting function: .a3d file animation. Slices along z-axis (z-axis is normal to slicing planes)
	"""
	data = read_data(path) # (x, z, y)
	fig = plt.figure(figsize = (16, 16))
	ax = fig.add_subplot(1, 1, 1)

	def animate(i):
		im = ax.imshow(data[:, i, :].transpose(), cmap = 'viridis')
		return [im]

	return animation.FuncAnimation(fig, animate, frames = range(0, data.shape[1]), interval = 200, blit = True)


def main():
	"""
	Main entry point for execution
	"""
	# Set up formatting for the movie files
	Writer = animation.writers['ffmpeg']
	writer = Writer(fps = 15, metadata = dict(artist = 'Me'), bitrate = 1800)

	# Create the destination directory, if necessary
	if not isdir(dirname(ani_file_out)):
		os.makedirs(dirname(ani_file_out), exist_ok = True)

	if ext_no_dot == "aps" or ext_no_dot == "a3daps":
		ani = plot_image_aps_or_a3daps(im_fpath)
		ani.save(ani_file_out, writer = writer)
	elif ext_no_dot == "a3d":
		pre = splitext(ani_file_out)[0]
		ext = splitext(ani_file_out)[1]
		ani_file_out_ver = pre + "_ver" + ext
		ani_file_out_hor = pre + "_hor" + ext
		ani_file_out_dep = pre + "_dep" + ext

		ani = plot_image_a3d_vertical_slices(im_fpath)
		ani.save(ani_file_out_ver, writer = writer)

		ani = plot_image_a3d_horizontal_slices(im_fpath)
		ani.save(ani_file_out_hor, writer = writer)

		ani = plot_image_a3d_depth_slices(im_fpath)
		ani.save(ani_file_out_dep, writer = writer)


if __name__ == '__main__':
	main()