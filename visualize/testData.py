from DataDistort import DataDistort
import numpy as np

test1 = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9,10,11,12],[13,14,15,16]], np.int32)

test2 = np.array([[[1, 2, 3, 4], [5, 6, 7, 8], [9,10,11,12],[13,14,15,16]],[[1, 2, 3, 4], [5, 6, 7, 8], [9,10,11,12],[13,14,15,16]],[[1, 2, 3, 4], [5, 6, 7, 8], [9,10,11,12],[13,14,15,16]],[[1, 2, 3, 4], [5, 6, 7, 8], [9,10,11,12],[13,14,15,16]]], np.int32)


#print(np.matrix(test1))

for i in range(len(test2)):
	print(np.matrix(test2[i]))

d = DataDistort()

shifted1 = d.random_shift_3d(test2,4,4,4)
shifted2 = d.random_shift_3d(test2,4,4,4)
shifted3 = d.random_shift_3d(test2,4,4,4)

for i in range(len(shifted1)):
	print(np.matrix(shifted1[i]))
for i in range(len(shifted2)):
	print(np.matrix(shifted2[i]))
for i in range(len(shifted3)):
	print(np.matrix(shifted3[i]))

