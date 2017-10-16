import sys
sys.path.insert(0, '../visualize/')

def main(argv):
	if len(argv) != 3:
		print("Please input 2 files to combine.")
	else:
		file1 = open(argv[1], "r")
		file1_points = file1.readlines()
		file1_points = [x.rstrip('\n') for x in file1_points]
		file1.close() 

		file2 = open(argv[2], "r")
		depth1 = file2.readline().split('\t')[1]
		file2.readline()
		depth2 = file2.readline().split('\t')[1]

		depths = [depth1,depth2]
		file2.close()

		output = open(argv[1].split('_')[0] + "_" + argv[1].split('_')[2].split(".")[0] + "_threatcube" + ".txt", "w")
		for d in depths:
			for l in file1_points:
				output.write(l + "	" + d)
		output.close()


if __name__ == "__main__": main(sys.argv)
