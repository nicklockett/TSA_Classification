import sys


def main(argv):
	if len(argv) != 3:
		print("please intput 2 files to combine")
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

		output = open(argv[1].split('_')[0] + "_threat" + argv[2].split("_")[1] + ".txt", "w")
		for d in depths:
			for l in file1_points:
				output.write(l + "	" + d + "\n")
		output.close()


if __name__ == "__main__": main(sys.argv)
