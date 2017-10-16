import sys
sys.path.insert(0, '../visualize/')
from classes import *


def main(argv):
	if len(argv) != 2:
		print("Please input the .a3d file that you will label.")
	else:
		bs = BodyScan(argv[1])
		filename_and_path = bs.create_max_projection("projections/")
		print("Projection created. File located at: " + filename_and_path)


if __name__ == "__main__": main(sys.argv)
