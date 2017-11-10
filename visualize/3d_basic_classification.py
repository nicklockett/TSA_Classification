from classes import *
from dataExtraction import *
from random import shuffle
from sklearn import svm, metrics, tree, naive_bayes
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AffinityPropagation

import matplotlib.pyplot as plt

block_size = 44

sc = SupervisedClassifier('../../stage1_labels.csv')

# need to ingest and iterate through multiple bodies
image_path_list = [
	"../precise_labeling/a3d/1e4a14d2e1eb381b773446de1c0c0b7e.a3d",
	#"../precise_labeling/a3d/1e5f7a00ff4f02b0de0ea97139b6c92e.a3d"
]

### Formating Data ###

# right now we're just throwing all the blocks from all images into the same list which is fed to the classifier
uniform_blocks = []

for file_path in image_path_list:
	
	bs = BodyScan(file_path)
	bsg = BlockStreamGenerator(bs, sc, blockSize = 44)
	block_list = bsg.generate3DBlockStreamHandLabeled()
	
	
	for block in block_list:
		# checking to make sure it is a full block and not an edge case
		if block.data.shape == (block.n, block.n, block.n):
			uniform_blocks.append(block)
	
# shuffling so that the subset below doesn't bias the training data
shuffle(uniform_blocks)

# subset images into test and training data
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
"""
print("--- K-Means Output ---")
kmeans_classifier = KMeans(n_clusters=2, random_state=0).fit(final_data_train)
print("Done training.")

predicted = kmeans_classifier.predict(final_data_test)

# output results
print("Classification report for classifier %s:\n%s\n"
      % (kmeans_classifier, metrics.classification_report(final_labels_test, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(final_labels_test, predicted))

### AffinityPropagation ###
# WARNING: this is super memory intensive
print("--- AffinityPropagation Output ---")
affinity_classifier = AffinityPropagation(max_iter=50).fit(final_data_train)
print("Done training.")
predicted = affinity_classifier.predict(final_data_test)
# output results
print("Classification report for classifier %s:\n%s\n"
      % (affinity_classifier, metrics.classification_report(final_labels_test, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(final_labels_test, predicted))
"""

###### Supervised Learning ######
"""
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
"""

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

"""
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
"""