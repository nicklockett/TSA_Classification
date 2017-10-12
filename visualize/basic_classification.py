from classes import *
from dataExtraction import *
from random import shuffle
from sklearn import svm, metrics, tree, naive_bayes
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AffinityPropagation

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

training_data = uniform_blocks[:block_count/2]
print(len(training_data))
test_data = uniform_blocks[block_count/2:]
print(len(test_data))

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

###### Dimensionality Reduction ######

### PCA ###
"""
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
"""

###### Unsupervised Learning ######

### K-Means ###
print("--- K-Means Output ---")
kmeans_classifier = KMeans(n_clusters=2, random_state=0).fit(final_data_train)
print("Done training.")

predicted = kmeans_classifier.predict(final_data_test)

# output results
print("Classification report for classifier %s:\n%s\n"
      % (kmeans_classifier, metrics.classification_report(final_labels_test, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(final_labels_test, predicted))

###### Supervised Learning ######

### SVM ###
print("--- SVM Output ---")

# create the classifier
svm_classifier = svm.SVC(gamma=0.001)

# we learn on the training data
svm_classifier.fit(final_data_train, final_labels)
print("Done training.")

# now predict the threat
predicted = svm_classifier.predict(final_data_test)

# output results
print("Classification report for classifier %s:\n%s\n"
      % (svm_classifier, metrics.classification_report(final_labels_test, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(final_labels_test, predicted))


### Basic Decision Tree ###
print("--- Decision Tree Output ---")

tree_classifier = tree.DecisionTreeClassifier()

#learn
tree_classifier = tree_classifier.fit(final_data_train, final_labels)
print("Done training.")

# predict
predicted = tree_classifier.predict(final_data_test)

# output results
print("Classification report for classifier %s:\n%s\n"
      % (tree_classifier, metrics.classification_report(final_labels_test, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(final_labels_test, predicted))


### RandomForestClassifier ###

print("--- RandomForestClassifier Output ---")

forest_classifier = RandomForestClassifier(n_estimators=10)

#learn
forest_classifier = forest_classifier.fit(final_data_train, final_labels)
print("Done training.")

# predict
predicted = forest_classifier.predict(final_data_test)

# output results
print("Classification report for classifier %s:\n%s\n"
      % (forest_classifier, metrics.classification_report(final_labels_test, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(final_labels_test, predicted))

print("--- AdaBoostClassifier Output ---")

ada_classifier = AdaBoostClassifier(n_estimators=100)

#learn
ada_classifier = ada_classifier.fit(final_data_train, final_labels)
print("Done training.")

# predict
predicted = ada_classifier.predict(final_data_test)

# output results
print("Classification report for classifier %s:\n%s\n"
      % (ada_classifier, metrics.classification_report(final_labels_test, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(final_labels_test, predicted))


### Gaussian Naive Bayes ###
print("--- Gaussian Naive Bayes Output ---")

gnb_classifier = naive_bayes.GaussianNB()

gnb_classifier = gnb_classifier.fit(final_data_train, final_labels)
print("Done training.")

predicted = gnb_classifier.predict(final_data_test)

# output results
print("Classification report for classifier %s:\n%s\n"
      % (gnb_classifier, metrics.classification_report(final_labels_test, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(final_labels_test, predicted))

### Multinomial Naive Bayes ###
print("--- Multinomial Naive Bayes Output ---")

mnb_classifier = naive_bayes.MultinomialNB()

mnb_classifier = mnb_classifier.fit(final_data_train, final_labels)
print("Done training.")

predicted = mnb_classifier.predict(final_data_test)

# output results
print("Classification report for classifier %s:\n%s\n"
      % (mnb_classifier, metrics.classification_report(final_labels_test, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(final_labels_test, predicted))


### Basic Multi-Layer Perceptron ###

from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()  
# Don't cheat - fit only on training data
scaler.fit(final_data_train)  
scaled_train = scaler.transform(final_data_train)  
# apply same transformation to test data
scaled_test = scaler.transform(final_data_test)  

print("--- Basic Multi-Layer Perceptron Output ---")
from sklearn.neural_network import MLPClassifier


mlp_classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

mlp_classifier.fit(scaled_train, final_labels)
print("Done training.")

predicted = mlp_classifier.predict(scaled_test)

# output results
print("Classification report for classifier %s:\n%s\n"
      % (mlp_classifier, metrics.classification_report(final_labels_test, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(final_labels_test, predicted))