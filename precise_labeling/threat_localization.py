import sys
sys.path.insert(0, '../visualize/')

def main(argv):
	if len(argv) != 3:
		print("Please input the file id and region.")
	else:
		file1_directory = "xyfiles/"+argv[1]+"_xy_"+argv[2]+".txt"
		file2_directory = "xyfiles/"+argv[1]+"_xy_"+argv[2]+"_slice.txt"

		file1 = open(file1_directory, "r")
		file1_points = file1.readlines()
		file1_points = [x.rstrip('\n') for x in file1_points]
		file1.close() 

		file2 = open(file2_directory, "r")
		depth1 = file2.readline().split('\t')[1]
		file2.readline()
		depth2 = file2.readline().split('\t')[1]

		depths = [depth1,depth2]
		file2.close()

		output = open(file1_directory.split('_')[0] + "_" + file1_directory.split('_')[2].split(".")[0] + "_threatcube" + ".txt", "w")
		for d in depths:
			for l in file1_points:
				output.write(l + "	" + d)
		output.close()


if __name__ == "__main__": main(sys.argv)
