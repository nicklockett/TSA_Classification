import tensorflow as tf

# read in datastream


# batch size
batch_size = 16
block_size = 32
train_path=#'training_data'

#build dcnn
a = tf.truncated_normal([batch_size,block_size,block_size,block_size])
sess = tf.Session()
sess.run(tf.initialize_all_variables())
sess.run(tf.shape(a))
b=tf.reshape(a,[batch_size,block_size**3])
sess.run(tf.shape(b))
classes = ['threat', 'nonthreat']
num_classes = len(classes)

# validation split
validation_size = 0.2

data = dataset.read_train_sets(train_path, img_size, classes, validation_size=validation_size)

tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))

def create_convolutional_layer(input,
               num_input_channels, 
               conv_filter_size,        
               num_filters):  
    
    ## We shall define the weights that will be trained using create_weights function.
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
    ## We create biases using the create_biases function. These are also trained.
    biases = create_biases(num_filters)

    ## Creating the convolutional layer
    layer = tf.nn.conv2d(input=input,
                     filter=weights,
                     strides=[1, 1, 1, 1],
                     padding='SAME')

    layer += biases

    ## We shall be using max-pooling.  
    layer = tf.nn.max_pool(value=layer,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME')
    ## Output of pooling is fed to Relu which is the activation function for us.
    layer = tf.nn.relu(layer)

    return layer


#train dcnn

#test dcnn and output results