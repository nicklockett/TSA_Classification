from classes import *
from dataExtraction import *
from random import shuffle
from sklearn import svm, metrics, tree, naive_bayes
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AffinityPropagation

from basic_cnn import extract2DDataSet

import matplotlib.pyplot as plt

block_size = 44
# Set segmentNumber
segmentNumber = 0.8

sc = SupervisedClassifier('../../stage1_labels.csv')

# need to ingest and iterate through multiple bodies
image_path_list = [
	"../precise_labeling/a3d/1e4a14d2e1eb381b773446de1c0c0b7e.a3d",
	#"../precise_labeling/a3d/1e5f7a00ff4f02b0de0ea97139b6c92e.a3d"
]

# Load training and eval data
(train_data, train_labels, eval_data, eval_labels) = extract2DDataSet(image_path_list, block_size, segmentNumber, sc)
flattened_train = []
for item in train_data:
	flattened_train.append(item.flatten())
train_data = flattened_train

flattened_eval = []
for item in eval_data:
	flattened_eval.append(item.flatten())
eval_data = flattened_eval



###### Dimensionality Reduction ######

### PCA ###
"""
print("--- PCA Output ---")

print(len(eval_data + train_data))
pca = PCA(n_components=3)
pca.fit(eval_data + train_data)

print(pca.components_.shape)
print("Explained variance per component:")
print(pca.explained_variance_ratio_)  

# plot first (x) and second (y) components
plt.scatter(pca.components_[0], pca.components_[1])
plt.show()


###### Unsupervised Learning ######
"""
### K-Means ###
"""
print("--- K-Means Output ---")
kmeans_classifier = KMeans(n_clusters=2, random_state=0).fit(train_data)
print("Done training.")

predicted = kmeans_classifier.predict(eval_data)

# output results
print("Classification report for classifier %s:\n%s\n"
      % (kmeans_classifier, metrics.classification_report(eval_labels, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(eval_labels, predicted))

### AffinityPropagation ###
# WARNING: this is super memory intensive
print("--- AffinityPropagation Output ---")
affinity_classifier = AffinityPropagation(max_iter=50).fit(train_data)
print("Done training.")

predicted = affinity_classifier.predict(eval_data)

# output results
print("Classification report for classifier %s:\n%s\n"
      % (affinity_classifier, metrics.classification_report(eval_labels, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(eval_labels, predicted))

###### Supervised Learning ######
"""
### SVM ###
print("--- SVM Output ---")

# create the classifier
svm_classifier = svm.SVC(gamma=0.001)

# we learn on the training data
svm_classifier.fit(train_data, train_labels)
print("Done training.")

# now predict the threat
predicted = svm_classifier.predict(eval_data)

# output results
print("Classification report for classifier %s:\n%s\n"
      % (svm_classifier, metrics.classification_report(eval_labels, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(eval_labels, predicted))

### Basic Decision Tree ###
print("--- Decision Tree Output ---")

tree_classifier = tree.DecisionTreeClassifier()

#learn
tree_classifier = tree_classifier.fit(train_data, train_labels)
print("Done training.")

# predict
predicted = tree_classifier.predict(eval_data)

# output results
print("Classification report for classifier %s:\n%s\n"
      % (tree_classifier, metrics.classification_report(eval_labels, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(eval_labels, predicted))

### RandomForestClassifier ###

print("--- RandomForestClassifier Output ---")

forest_classifier = RandomForestClassifier(n_estimators=10)

#learn
forest_classifier = forest_classifier.fit(train_data, train_labels)
print("Done training.")

# predict
predicted = forest_classifier.predict(eval_data)

# output results
print("Classification report for classifier %s:\n%s\n"
      % (forest_classifier, metrics.classification_report(eval_labels, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(eval_labels, predicted))

"""
print("--- AdaBoostClassifier Output ---")

ada_classifier = AdaBoostClassifier(n_estimators=100)

#learn
ada_classifier = ada_classifier.fit(train_data, train_labels)
print("Done training.")

# predict
predicted = ada_classifier.predict(eval_data)

# output results
print("Classification report for classifier %s:\n%s\n"
      % (ada_classifier, metrics.classification_report(eval_labels, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(eval_labels, predicted))


### Gaussian Naive Bayes ###
print("--- Gaussian Naive Bayes Output ---")

gnb_classifier = naive_bayes.GaussianNB()

gnb_classifier = gnb_classifier.fit(train_data, train_labels)
print("Done training.")

predicted = gnb_classifier.predict(eval_data)

# output results
print("Classification report for classifier %s:\n%s\n"
      % (gnb_classifier, metrics.classification_report(eval_labels, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(eval_labels, predicted))

### Multinomial Naive Bayes ###
print("--- Multinomial Naive Bayes Output ---")

mnb_classifier = naive_bayes.MultinomialNB()

mnb_classifier = mnb_classifier.fit(train_data, train_labels)
print("Done training.")

predicted = mnb_classifier.predict(eval_data)

# output results
print("Classification report for classifier %s:\n%s\n"
      % (mnb_classifier, metrics.classification_report(eval_labels, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(eval_labels, predicted))


### Basic Multi-Layer Perceptron ###

from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()  
# Don't cheat - fit only on training data
scaler.fit(train_data)  
scaled_train = scaler.transform(train_data)  
# apply same transformation to test data
scaled_test = scaler.transform(eval_data)  

print("--- Basic Multi-Layer Perceptron Output ---")
from sklearn.neural_network import MLPClassifier


mlp_classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

mlp_classifier.fit(scaled_train, train_labels)
print("Done training.")

predicted = mlp_classifier.predict(scaled_test)

# output results
print("Classification report for classifier %s:\n%s\n"
      % (mlp_classifier, metrics.classification_report(eval_labels, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(eval_labels, predicted))

"""
