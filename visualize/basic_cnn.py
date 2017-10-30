from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
from classes import *
from dataExtraction import *
from random import shuffle

tf.logging.set_verbosity(tf.logging.INFO)
block_size = 28

def main(unused_argv):

	# Set up tools needed
	sc = SupervisedClassifier('../../stage1_labels.csv')

	# Set image list for use
	image_path_list = []
	image_path_list.append("../precise_labeling/a3d/0e34d284cb190a79e3316d1926061bc3.a3d")

	# Set segmentNumber
	segmentNumber = 0.8

	# Load training and eval data
	(train_data, train_labels, eval_data, eval_labels) = extract2DDataSet(image_path_list, block_size, segmentNumber, sc)

	# Create the Estimator
	mnist_classifier = tf.estimator.Estimator(
	    model_fn=cnn_model_fn, model_dir="../cnn_model_output")

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

def extract2DDataSet(image_path_list, block_size, segmentNumber, supervisedClassifier):
    """This method returns the 2D training data, training labels, 
    testing data, and testing labels for a particular data set"""

    data_label_stream = []

    print (image_path_list)

    print ('looking for segment ', segmentNumber)

    for image_path in image_path_list:
    	print('about to create a body scan with filepath ', image_path)
        bs = BodyScan(image_path)
        bsg = BlockStreamGenerator(bs, supervisedClassifier, blockSize = block_size)
        block_list = bsg.generate2DBlockStreamHandLabeled()
        
        for block in block_list:
        	print ('block shape ', block[0].shape)
        	print (block[0].shape == (block_size, block_size))
        	if block[0].shape == (block_size, block_size):
        		if(segmentNumber == block[1]):
        			data_label_stream.append((block[0], int(block[2])))

    print('total data length: ',len(data_label_stream))

    shuffle(data_label_stream)

    data_stream = []
    label_stream = []

    for data_label in data_label_stream:
    	data_stream.append(data_label[0])
    	label_stream.append(data_label[1])

    print ('data stream length: ', len(data_stream))
    print ('label stream length: ', len(label_stream))

    print('type data stream: ',type(data_stream[0]))
    print('type label stream: ',type(label_stream[0]))

    # Determine indexing length
    trainingLength = int(len(data_label_stream)/2)

    print ('training length: ', trainingLength)

    # Index proper sizes
    trainingData = data_stream[:trainingLength]
    trainingLabels = label_stream[:trainingLength]
    testingData = data_stream[trainingLength:]
    testingLabels = label_stream[trainingLength:]

    # Convert to numpy arrays
    trainingData = np.array(trainingData)
    trainingLabels = np.array(trainingLabels)
    testingData = np.array(testingData)
    testingLabels = np.array(testingLabels)

    return (trainingData, trainingLabels, testingData, testingLabels)

# Our application logic will be added here
def cnn_model_fn(features, labels, mode):

	block_size = 28

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
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
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

if __name__ == "__main__":
	tf.app.run(main=main)