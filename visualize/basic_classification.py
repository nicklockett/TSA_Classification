from classes import *
from dataExtraction import *
from random import shuffle
from sklearn import svm, metrics
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# need to ingest and iterate through multiple bodies
images_list = [
	#"../data/data/a3d/f2c1d30f352f6b5ab8dd5da31f85ee1d.a3d",
	#"../data/data/a3d/f35a31e8b666ba97841c98ae6a26f3ef.a3d",
	#"../data/data/a3d/f412f718c4ef81b6a7ce4b46651596ce.a3d",
	#"../data/data/a3d/f5283ffe7e484ffc640ebcf62b534f7b.a3d",
	#"../data/data/a3d/f5fed2604c69f028efba9c92459abe79.a3d",
	"../data/data/a3d/f6303b38942d876f160302be6d2c34eb.a3d",
	"../data/data/a3d/f92d8566ff9460451ab42093098a0efe.a3d",
	"../data/data/a3d/f96fe81c61c951792a43cb4851ca3829.a3d",
	"../data/data/a3d/faa3d6f358099ee2b091a5b87feca844.a3d",
	"../data/data/a3d/fac0f28db837e3ae9f965560718674c6.a3d",
	"../data/data/a3d/fae2676a3d4bd35b0b7088fad9f2e554.a3d",
	"../data/data/a3d/fdb996a779e5d65d043eaa160ec2f09f.a3d"
	]

### Formating Data ###

# right now we're just throwing all the blocks from all images into the same list which is fed to the classifier
uniform_blocks = []
sc = SupervisedClassifier('../data/data/stage1_labels.csv')

for file_path in images_list:
	
	bs = BodyScan(file_path)
	bsg = BlockStreamGenerator(bs, sc)
	block_list = bsg.generateStream()
	
	
	for block in block_list:
		# checking to make sure it is a full block and not an edge case
		if block.data.shape == (block.n, block.n, block.n):
			uniform_blocks.append(block)
	
# shuffling so that the subset below doesn't bias the training data
shuffle(uniform_blocks)

# subset image into test and training data
# TODO would this will be more effective to subset whole images into test vs training?
block_count = len(uniform_blocks)
training_data = uniform_blocks[:block_count/4]
test_data = uniform_blocks[block_count*3/4:]

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



