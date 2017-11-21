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

width = 40
height = 40
depth = 40
nLabel = 2



def main(unused_argv):

    # Set up tools needed
    sc = SupervisedClassifier('../../stage1_labels.csv')

    # Set image list for use
    image_path_list = ["../precise_labeling/a3d/0e34d284cb190a79e3316d1926061bc3.a3d",
"../precise_labeling/a3d/1636ba745a6fc4d97dba1d27825de2b0.a3d",
"../precise_labeling/a3d/011516ab0eca7cad7f5257672ddde70e.a3d",
"../precise_labeling/a3d/0d10b14405f0443be67a75554da778a0.a3d",
"../precise_labeling/a3d/2f5c066720d997f33453dc491141bc70.a3d",
"../precise_labeling/a3d/0fdad88d401b09d417ffbc490640d9e2.a3d",
"../precise_labeling/a3d/098f5cfcf6faefd3011a94719cb03dc5.a3d",
"../precise_labeling/a3d/0240c8f1e89e855dcd8f1fa6b1e2b944.a3d",
"../precise_labeling/a3d/3c0668db35915783be0af87b9fa53317.a3d",
"../precise_labeling/a3d/41ed36bdc93a2ff5519c76f263ab1a88.a3d",
"../precise_labeling/a3d/04b32b70b4ab15cad85d43e3b5359239.a3d",
"../precise_labeling/a3d/42181583618ce4bbfbc0c4c300108bf5.a3d",
"../precise_labeling/a3d/0e34d284cb190a79e3316d1926061bc3.a3d",
"../precise_labeling/a3d/0043db5e8c819bffc15261b1f1ac5e42.a3d",
"../precise_labeling/a3d/3eda71e0fd6f0c18c1fa43371c4212e4.a3d",
"../precise_labeling/a3d/052117021fc1396db6bae78ffe923ee4.a3d",
"../precise_labeling/a3d/11f9ae01877f6c0becf49c709eddb8cb.a3d"]

    # Set segmentNumber
    segmentNumber = 0.8

    # Load training and eval data
    (train_data, train_labels, eval_data, eval_labels) = extract3DDataSet(image_path_list, block_size, segmentNumber, sc)

    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="../cnn_model_output_4")

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
        steps=20000, # QUESTION: why do we need so many steps here... 
        hooks=[logging_hook])

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)

def extract3DDataSet(image_path_list, block_size, segmentNumber, supervisedClassifier):
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
            if block[0].shape == (block_size, block_size):
                #if(segmentNumber == block[1]):
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

    print('labels: ',label_stream)
    # Determine indexing length
    trainingLength = int(len(data_label_stream)/2)

    print ('training length: ', trainingLength)

    # Index proper sizes
    trainingData = data_stream[:trainingLength]
    trainingLabels = label_stream[:trainingLength]
    testingData = data_stream[trainingLength:]
    testingLabels = label_stream[trainingLength:]

    print('training labels: ',trainingLabels)
    print('testing labels: ',testingLabels)

    # Convert to numpy arrays
    trainingData = np.array(trainingData)
    trainingLabels = np.array(trainingLabels)
    testingData = np.array(testingData)
    testingLabels = np.array(testingLabels)

    return (trainingData, trainingLabels, testingData, testingLabels)

# Our application logic will be added here
def cnn_model_fn(features, labels, mode):


    ## First Convolutional Layer
    # Conv then Max-pooling. 1st layer will have 32 features for each 5x5 patch. (1 feature -> 32 features)
    W_conv1 = weight_variable([5, 5, 5, 1, 32])  # shape of weight tensor = [5,5,1,32]
    b_conv1 = bias_variable([32])  # bias vector for each output channel. = [32]

    # Reshape 'x' to a 4D tensor (2nd dim=image width, 3rd dim=image height, 4th dim=nColorChannel)
    x_image = tf.reshape(x, [-1,width,height,depth,1]) # [-1,28,28,1]
    print(x_image.get_shape) # (?, 256, 256, 40, 1)  # -> output image: 28x28 x1

    # x_image * weight tensor + bias -> apply ReLU -> apply max-pool
    h_conv1 = tf.nn.relu(conv3d(x_image, W_conv1) + b_conv1)  # conv2d, ReLU(x_image * weight + bias)
    print(h_conv1.get_shape) # (?, 256, 256, 40, 32)  # -> output image: 28x28 x32
    h_pool1 = max_pool_2x2(h_conv1)  # apply max-pool 
    print(h_pool1.get_shape) # (?, 128, 128, 20, 32)  # -> output image: 14x14 x32


    ## Second Convolutional Layer
    # Conv then Max-pooling. 2nd layer will have 64 features for each 5x5 patch. (32 features -> 64 features)
    W_conv2 = weight_variable([5, 5, 5, 32, 64]) # [5, 5, 32, 64]
    b_conv2 = bias_variable([64]) # [64]

    h_conv2 = tf.nn.relu(conv3d(h_pool1, W_conv2) + b_conv2)  # conv2d, .ReLU(x_image * weight + bias)
    print(h_conv2.get_shape) # (?, 128, 128, 20, 64)  # -> output image: 14x14 x64
    h_pool2 = max_pool_2x2(h_conv2)  # apply max-pool 
    print(h_pool2.get_shape) # (?, 64, 64, 10, 64)    # -> output image: 7x7 x64


    ## Densely Connected Layer (or fully-connected layer)
    # fully-connected layer with 1024 neurons to process on the entire image
    W_fc1 = weight_variable([16*16*3*64, 1024])  # [7*7*64, 1024]
    b_fc1 = bias_variable([1024]) # [1024]]

    h_pool2_flat = tf.reshape(h_pool2, [-1, 16*16*3*64])  # -> output image: [-1, 7*7*64] = 3136
    print(h_pool2_flat.get_shape)  # (?, 2621440)
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  # ReLU(h_pool2_flat x weight + bias)
    print(h_fc1.get_shape) # (?, 1024)  # -> output: 1024

    ## Dropout (to reduce overfitting; useful when training very large neural network)
    # We will turn on dropout during training & turn off during testing
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    print(h_fc1_drop.get_shape)  # -> output: 1024

    ## Readout Layer
    W_fc2 = weight_variable([1024, nLabel]) # [1024, 10]
    b_fc2 = bias_variable([nLabel]) # [10]




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



## Weight Initialization
# Create lots of weights and biases & Initialize with a small positive number as we will use ReLU
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

## Convolution and Pooling
# Convolution here: stride=1, zero-padded -> output size = input size
def conv3d(x, W):
  return tf.nn.conv3d(x, W, strides=[10, 10, 10, 10, 10], padding='SAME') # conv2d, [1, 1, 1, 1]

# Pooling: max pooling over 2x2 blocks
def max_pool_2x2(x):  # tf.nn.max_pool. ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1]
  return tf.nn.max_pool3d(x, ksize=[1, 4, 4, 4, 1], strides=[1, 4, 4, 4, 1], padding='SAME')

if __name__ == "__main__":
    tf.app.run(main=main)







# A simple CNN to predict certain characteristics of the human subject from MRI images.
# 3d convolution is used in each layer.
# Reference: https://www.tensorflow.org/get_started/mnist/pros, http://blog.naver.com/kjpark79/220783765651
# Adjust needed for your dataset e.g., max pooling, convolution parameters, training_step, batch size, etc

# Start TensorFlow InteractiveSession
import tensorflow as tf
sess = tf.InteractiveSession()

# Placeholders (MNIST image:28x28pixels=784, label=10)
x = tf.placeholder(tf.float32, shape=[None, width*height*depth]) # [None, 28*28]
y_ = tf.placeholder(tf.float32, shape=[None, nLabel])  # [None, 10]





y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
print(y_conv.get_shape)  # -> output: 10

## Train and Evaluate the Model
# set up for optimization (optimizer:ADAM)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)  # 1e-4
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.global_variables_initializer())

# Include keep_prob in feed_dict to control dropout rate.
for i in range(100):
    batch = get_data_MRI(sess,'train',20)
    # Logging every 100th iteration in the training process.
    if i%5 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

# Evaulate our accuracy on the test data
testset = get_data_MRI(sess,'test',30)
print("test accuracy %g"%accuracy.eval(feed_dict={x: testset[0], y_: teseset[1], keep_prob: 1.0}))
