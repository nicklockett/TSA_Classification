from classes import *
from dataExtraction import *
from random import shuffle
from sklearn import svm, metrics
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# need to ingest and iterate through multiple bodies
images_list = [
	"../data/fdb996a779e5d65d043eaa160ec2f09f.a3d",
	]

### Formating Data ###

# right now we're just throwing all the blocks from all images into the same list which is fed to the classifier
uniform_blocks = []
sc = SupervisedClassifier('../../stage1_labels.csv')

for file_path in images_list:

	bs = BodyScan(file_path)
	bsg = BlockStreamGenerator(bs, sc)
	block_list = bsg.generateStream()

	for block in block_list:
		# checking to make sure it is a full block and not an edge case
		if block.data.shape == (block.n,block.n,block.n):
			uniform_blocks.append(block)

# shuffling so that the subset below doesn't bias the training data
shuffle(uniform_blocks)

# subset image into test and training data
# TODO would this will be more effective to subset whole images into test vs training?
block_count = len(uniform_blocks)
training_data = uniform_blocks[:block_count/10]
test_data = uniform_blocks[block_count*9/10:]

# flatten the blocks into a vector

# training data
for block in training_data:
	block.data = block.data.flatten()

final_data_train = []
for block in training_data:
	final_data_train.append(block.data)

final_labels = []
for block in training_data:
	final_labels.append(block.threat)

# test data
for block in test_data:
	block.data = block.data.flatten()

final_data_test = []
for block in test_data:
	final_data_test.append(block.data)

final_labels_test = []
for block in test_data:
	final_labels_test.append(block.threat)


### SVM ###
print("--- SVM Output ---")

# create the classifier
classifier = svm.SVC(gamma=0.001)

# we learn on the training data
classifier.fit(final_data_train, final_labels)
print("Done training.")

# now predict the threat
predicted = classifier.predict(final_data_test)

# output results
print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(final_labels_test, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(final_labels_test, predicted))


### PCA ###
print("--- PCA Output ---")

print(len(final_data_test + final_data_train))
pca = PCA(n_components=3)
pca.fit(final_data_test + final_data_train)

print(pca.components_.shape)
print("Explained variance per component:")
print(pca.explained_variance_ratio_)  

# plot first (x) and second (y) components
plt.scatter(pca.components_[0], pca.components_[1])
plt.show()



