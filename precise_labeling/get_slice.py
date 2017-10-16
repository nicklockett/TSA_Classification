import sys
sys.path.insert(0, '../visualize/')
from classes import *


def main(argv):
	if len(argv) != 2:
		print("Please input the xy coordinates file.")
	else:
		file1 = open(argv[1], "r")
		first_y = int(file1.readline().split('\t')[1])
		file1.readline()
		third_y = int(file1.readline().split('\t')[1])
		nthSlice = int(660 - (first_y + third_y)/2)
		file1.close() 

		a3d_path = "a3d/" + argv[1].split('_')[0].split("/")[1] + ".a3d"
		bs = BodyScan(a3d_path)
		slc = bs.get_single_image(nthSlice)

		changed_path = "slices/" + argv[1].split("/")[1]
		fileName = changed_path.split('.')[0] + "_slice.png"
		bs.write_square_slice_to_img(slc,fileName)


if __name__ == "__main__": main(sys.argv)
