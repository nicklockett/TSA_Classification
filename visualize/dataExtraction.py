threshold = .5
blockSize = 8
shift = blockSize / 2


for b in Blocks:
	x = 0;
	y = 0;
	z = 0;
	for x in range(0, len(b), shift):
		for z in range(0, len(x), shift):
			for y in range(0, len(z)):
				if b[x][z][y] >= threshold:
					grabBlock
					break
			for y in range(len(z), 0, -1):
				if(b[x][z][y] >= threshold):
					grabBlock
					break


def grabBlock(self, x, z, y):

	return 