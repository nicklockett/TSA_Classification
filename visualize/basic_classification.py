from classes import *
from dataExtraction import *
from random import shuffle
from sklearn import svm, metrics, tree, naive_bayes
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AffinityPropagation

import matplotlib.pyplot as plt

# need to ingest and iterate through multiple bodies
"""
images_list = [
	#"../data/data/a3d/f2c1d30f352f6b5ab8dd5da31f85ee1d.a3d",
	#"../data/data/a3d/f35a31e8b666ba97841c98ae6a26f3ef.a3d",
	#"../data/data/a3d/f412f718c4ef81b6a7ce4b46651596ce.a3d",
	#"../data/data/a3d/f5283ffe7e484ffc640ebcf62b534f7b.a3d",
	#"../data/data/a3d/f5fed2604c69f028efba9c92459abe79.a3d",
	#"../data/data/a3d/f6303b38942d876f160302be6d2c34eb.a3d",
	#"../data/data/a3d/f92d8566ff9460451ab42093098a0efe.a3d",
	#"../data/data/a3d/f96fe81c61c951792a43cb4851ca3829.a3d",
	#"../data/data/a3d/faa3d6f358099ee2b091a5b87feca844.a3d",
	#"../data/data/a3d/fac0f28db837e3ae9f965560718674c6.a3d",
	#"../data/data/a3d/fae2676a3d4bd35b0b7088fad9f2e554.a3d",
	#"../data/data/a3d/fdb996a779e5d65d043eaa160ec2f09f.a3d"
"../precise_labeling/a3d/00360f79fd6e02781457eda48f85da90.a3d",
"../precise_labeling/a3d/01c08047f617de893bef104fb309203a.a3d",
"../precise_labeling/a3d/0397026df63bbc8fd88f9860c6e35b4a.a3d",
"../precise_labeling/a3d/0043db5e8c819bffc15261b1f1ac5e42.a3d",
"../precise_labeling/a3d/0240c8f1e89e855dcd8f1fa6b1e2b944.a3d",
"../precise_labeling/a3d/03a36512c2c6d71c33b3429b8b59494e.a3d",
"../precise_labeling/a3d/0050492f92e22eed3474ae3a6fc907fa.a3d",
"../precise_labeling/a3d/0322661ef29f9c81af295cf40f758469.a3d",
"../precise_labeling/a3d/04b32b70b4ab15cad85d43e3b5359239.a3d",
"../precise_labeling/a3d/006ec59fa59dd80a64c85347eef810c7.a3d",
"../precise_labeling/a3d/0367394485447c1c3485359ba71f52cb.a3d",
"../precise_labeling/a3d/05709d5e54f8fdc77fe233cf7df78b81.a3d",
"../precise_labeling/a3d/011516ab0eca7cad7f5257672ddde70e.a3d",
"../precise_labeling/a3d/037024e4a7122e10546ebc41859c6833.a3d",
"../precise_labeling/a3d/01941f33fd090ae5df8c95992c027862.a3d",
"../precise_labeling/a3d/038d648c2f29cb0f945c865be25e32e9.a3d"
	]"""
images_list = [
	"../precise_labeling/a3d/1e4a14d2e1eb381b773446de1c0c0b7e.a3d"
]

### Formating Data ###

# right now we're just throwing all the blocks from all images into the same list which is fed to the classifier
uniform_blocks = []
sc = SupervisedClassifier('../data/data/stage1_labels.csv')

for file_path in images_list:
	print(file_path)
	bs = BodyScan(file_path)
	print('made body scan!')
	bsg = BlockStreamGenerator(bs, sc, blockSize = 36)
	#block_list = bsg.generateStream()
	block_list = bsg.generateStreamHandLabeled()
	print('generated blocks!')
	
	
	for block in block_list:
		# checking to make sure it is a full block and not an edge case
		if block.data.shape == (block.n, block.n, block.n):
			uniform_blocks.append(block)
	
# shuffling so that the subset below doesn't bias the training data
shuffle(uniform_blocks)

# subset image into test and training data
# TODO would this will be more effective to subset whole images into test vs training?
block_count = len(uniform_blocks)

training_data = uniform_blocks[:int(block_count/2)]
print('training data: ', len(training_data))
test_data = uniform_blocks[int(block_count/2):]
print('test data: ', len(test_data))

# flatten the blocks into a vector

# training data
for block in training_data:
	block.data = block.data.flatten()
print('flattened training data')

final_data_train = []
for block in training_data:
	final_data_train.append(block.data)

print('created final data training set')

final_labels = []
for block in training_data:
	final_labels.append(block.threat)

print('appended final labels to training data')

# test data
for block in test_data:
	block.data = block.data.flatten()

print('flattened test data')

final_data_test = []
for block in test_data:
	final_data_test.append(block.data)

print('created final test data set')

final_labels_test = []
for block in test_data:
	final_labels_test.append(block.threat)

print('appended final labels to test data')

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

"""
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

"""
