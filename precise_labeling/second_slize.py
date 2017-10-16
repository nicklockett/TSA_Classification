import sys
from classes import *


def main(argv):
	if len(argv) != 3:
		print("please intput the first xy coordinates file and the region containting the violation")
	else:
		file1 = open(argv[1], "r")
		first_y = int(file1.readline().split('\t')[1])
		print(first_y)
		file1.readline()
		third_y = int(file1.readline().split('\t')[1])
		print(third_y)
		nthSlice = 660 - (first_y + third_y)/2
		print(nthSlice)
		file1.close() 

		bs = BodyScan(argv[1].split('_')[0]+".a3d")
		slc = bs.get_single_image(nthSlice)

		fileName = argv[1].split('.')[0] + "_slice.png"
		bs.write_square_slice_to_img(slc,fileName)


if __name__ == "__main__": main(sys.argv)
