from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
from classes import *
from dataExtraction import *
from tf_dataset_creator import *
from random import shuffle

tf.logging.set_verbosity(tf.logging.INFO)
block_size = 50

def main(unused_argv):

	# Set up tools needed
	sc = SupervisedClassifier('../../stage1_labels.csv')


	# Set segmentNumber
	segmentNumber = -1

	# Load training and eval data
	dataCreator = TensorFlowDataSetCreator(sc)
	dataset = dataCreator.CreateTensorFlowDataSetFromBlockStream(block_size = block_size, augment = True)
	train_data = dataset.getTrainingData()
	train_labels = dataset.getTrainingLabels()
	eval_data = dataset.getTestingData()
	eval_labels = dataset.getTestingLabels()

	# Create the Estimator
	mnist_classifier = tf.estimator.Estimator(
	    model_fn=cnn_model_fn, model_dir="../cnn_model_output_7")

	# Set up logging for predictions
	tensors_to_log = {"probabilities": "softmax_tensor"}
	logging_hook = tf.train.LoggingTensorHook(
	tensors=tensors_to_log, every_n_iter=50)

	# Train the model
	train_input_fn = tf.estimator.inputs.numpy_input_fn(
	    x={"x": train_data},
	    y=train_labels,
	    batch_size=100, 
	    num_epochs=None,
	    shuffle=True)
	mnist_classifier.train(
	    input_fn=train_input_fn,
	    steps=20000,
	    hooks=[logging_hook])

	# Evaluate the model and print results
	eval_input_fn = tf.estimator.inputs.numpy_input_fn(
	    x={"x": eval_data},
	    y=eval_labels,
	    num_epochs=1,
	    shuffle=False)
	eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
	print(eval_results)

# Our application logic will be added here
def cnn_model_fn(features, labels, mode):

	"""Model function for CNN."""
	# Input Layer
	input_layer = tf.reshape(features["x"], [-1, block_size, block_size, 1])

	# Convolutional Layer #1
	conv1 = tf.layers.conv2d(
	  inputs=input_layer,
	  filters=32,
	  kernel_size=[5, 5],
	  padding="same",
	  activation=tf.nn.relu)

	# Pooling Layer #1
	pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
	# conv1 has shape [batch_size, block_size, block_size, 32]
	# pool1 has shape [batch_size, block_size/2, block_size/2, 32] b/c pool size reduces by 50% here

	# Convolutional Layer #2 and Pooling Layer #2
	conv2 = tf.layers.conv2d(
	  inputs=pool1,
	  filters=64,
	  kernel_size=[5, 5],
	  padding="same",
	  activation=tf.nn.relu)

	# Pooling Layer #2
	pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
	# conv2 has a shape of [batch_size, blocksize/2, blocksize/2, 64]
	# pool2 has a shape of [batch_size, blocksize/4, blocksize/4, 64]

	# Dense Layer
	pool2_flat = tf.reshape(pool2, [-1, int(block_size/4) * int(block_size/4) * 64])
	# pool2_flat has shape [batch_size, (blocksize/4) * (blocksize/4) * 64]

	# QUESTION: not sure why we are using 1024 unites here
	# Create dense layer now that CNN features have been built
	dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
	
	# Apply drop out if we are in training mode
	dropout = tf.layers.dropout(
	  inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
	# dropout has shape [batch_size, 1024]

	# Logits Layer, we specify one unit for each target class
	num_classes = 2 # either threat or no threat 
	logits = tf.layers.dense(inputs=dropout, units=num_classes) # TODO: make sure the units here are correct
	# logits gives our predictions as raw values in a [batch_size,2] dimensional vector
	# basically gives predicted class and probability of confidence 

	predictions = {
	  # Generate predictions (for PREDICT and EVAL mode)
	  "classes": tf.argmax(input=logits, axis=1), # gets us the biggest of the predictions
	  # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
	  # `logging_hook`.
	  "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
	}

	# If we are in prediction mode, return all of our prediction as a dict
	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

	# If in training mode, we need to figure out how we did 

	# Calculate Loss (for both TRAIN and EVAL modes)
	onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=num_classes)
	loss = tf.losses.softmax_cross_entropy(
	  onehot_labels=onehot_labels, logits=logits)

	# Configure the Training Op (for TRAIN mode)
	# QUESTION: why does this training happen later?
	if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001) # Was previously .001
		train_op = optimizer.minimize(
		    loss=loss,
		    global_step=tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

	# Add evaluation metrics (for EVAL mode)
	eval_metric_ops = {
	  "accuracy": tf.metrics.accuracy(
	      labels=labels, predictions=predictions["classes"])}
	return tf.estimator.EstimatorSpec(
	  mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

	# Add FP/FN metrics (for EVAL mode)
	predicted = tf.round(tf.nn.sigmoid(logits))
	actual = labels
	    
	# Count true positives, true negatives, false positives and false negatives.
	tp = tf.count_nonzero(predicted * actual)
	tn = tf.count_nonzero((predicted - 1) * (actual - 1))
	fp = tf.count_nonzero(predicted * (actual - 1))
	fn = tf.count_nonzero((predicted - 1) * actual)

	print ('tp: ', tp)
	print ('tn: ', tn)
	print ('fp: ', fp)
	print ('fn: ', fn)
	
	# Calculate accuracy, precision, recall and F1 score.
	accuracy = (tp + tn) / (tp + fp + fn + tn)
	precision = tp / (tp + fp)
	recall = tp / (tp + fn)
	fmeasure = (2 * precision * recall) / (precision + recall)

	# Add metrics to TensorBoard.    
	tf.summary.scalar('Accuracy', accuracy)
	tf.summary.scalar('Precision', precision)
	tf.summary.scalar('Recall', recall)
	tf.summary.scalar('f-measure', fmeasure)

	print ('Accuracy: ', accuracy)
	print ('Precision: ', precision)
	print ('Recall: ', recall)
	print ('f-measure: ', fmeasure)

if __name__ == "__main__":
	tf.app.run(main=main)
