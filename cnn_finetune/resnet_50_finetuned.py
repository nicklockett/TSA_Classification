# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
from keras.layers.normalization import BatchNormalization
from keras.utils.np_utils import to_categorical
from keras.models import Model
from keras import backend as K
from sklearn.metrics import log_loss
import sys
sys.path.append('../visualize')
from tf_dataset_creator import *
from classes import *
import cv2
import os
import scipy
#from load_cifar10 import load_cifar10_data

def identity_block(input_tensor, kernel_size, filters, stage, block):
    """
    The identity_block is the block that has no conv layer at shortcut
    Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    """

    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, 1, 1, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter2, kernel_size, kernel_size,
                      border_mode='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter3, 1, 1, name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = merge([x, input_tensor], mode='sum')
    x = Activation('relu')(x)
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """
    conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    """

    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, 1, 1, subsample=strides,
                      name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter2, kernel_size, kernel_size, border_mode='same',
                      name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter3, 1, 1, name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Convolution2D(nb_filter3, 1, 1, subsample=strides,
                             name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = merge([x, shortcut], mode='sum')
    x = Activation('relu')(x)
    return x

def resnet50_model(img_rows, img_cols, color_type = 1, num_classes = None):
    """
    Resnet 50 Model for Keras

    Model Schema is based on 
    https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py

    ImageNet Pretrained Weights 
    https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_th_dim_ordering_th_kernels.h5

    Parameters:
      img_rows, img_cols - resolution of inputs
      channel - 1 for grayscale, 3 for color 
      num_classes - number of class labels for our classification task
    """
    print 'color_type: ', color_type

    # Handle Dimension Ordering for different backends
    global bn_axis
    if K.image_dim_ordering() == 'tf':
      bn_axis = 3
      img_input = Input(shape=(img_rows, img_cols, color_type))
    else:
      bn_axis = 1
      img_input = Input(shape=(color_type, img_rows, img_cols))

    x = ZeroPadding2D((3, 3))(img_input)
    x = Convolution2D(64, 7, 7, subsample=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, channel, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, channel, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, channel, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, channel, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, channel, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, channel, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, channel, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, channel, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, channel, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, channel, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, channel, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, channel, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, channel, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, channel, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, channel, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, channel, [512, 512, 2048], stage=5, block='c')

    # Fully Connected Softmax Layer
    x_fc = AveragePooling2D((7, 7), name='avg_pool')(x)
    x_fc = Flatten()(x_fc)
    x_fc = Dense(1000, activation='softmax', name='fc1000')(x_fc)

    # Create model
    model = Model(img_input, x_fc)

    # Load ImageNet pre-trained data 
    if K.image_dim_ordering() == 'th':
      # Use pre-trained weights for Theano backend
      weights_path = 'imagenet_models/resnet50_weights_th_dim_ordering_th_kernels.h5'
    else:
      # Use pre-trained weights for Tensorflow backend
      weights_path = 'imagenet_models/resnet50_weights_tf_dim_ordering_tf_kernels.h5'

    model.load_weights(weights_path)

    # Truncate and replace softmax layer for transfer learning
    # Cannot use model.layers.pop() since model is not of Sequential() type
    # The method below works since pre-trained weights are stored in layers but not in the model
    x_newfc = AveragePooling2D((7, 7), name='avg_pool')(x)
    x_newfc = Flatten()(x_newfc)
    x_newfc = Dense(num_classes, activation='softmax', name='fc10')(x_newfc)

    # Create another model with our customized softmax
    model = Model(img_input, x_newfc)

    # Learning rate is changed to 0.001
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
  
    return model

def load_images_from_folder(max_folder, sum_folder, var_folder, resize):

    max_image_filenames = os.listdir(max_folder)
    sum_image_filenames = os.listdir(sum_folder)
    var_image_filenames = os.listdir(var_folder)

    training_length = len(max_image_filenames)/2
    testing_lenth = len(max_image_filenames) - training_length

    X_train = np.empty((training_length, resize, resize, 3))
    Y_train = np.empty(training_length)
    X_test = np.empty((testing_lenth, resize, resize, 3))
    Y_test = np.empty(testing_lenth)

    for index in range(0,len(max_image_filenames)):

        max_image_filename = max_image_filenames[index]
        sum_image_filename = sum_image_filenames[index]
        var_image_filename = var_image_filenames[index]

        file_id, channel_type, is_threat, region, x, y = max_image_filename.split("_")
        
        # read in the image
        max_array = scipy.misc.imread(os.path.join(max_folder,max_image_filename), mode = 'L')
        sum_array = scipy.misc.imread(os.path.join(sum_folder,sum_image_filename), mode = 'L')
        var_array = scipy.misc.imread(os.path.join(var_folder,var_image_filename), mode = 'L')

        # convert to arrray
        #max_array = np.asarray(max_image)
        #sum_array = np.asarray(sum_image)
        #var_array = np.asarray(var_image)

        #print 'max array shape', max_array.shape

        # resize the image
        Channeled_Data = np.zeros((resize,resize,3))
        data_channel_1 = scipy.misc.imresize(arr = max_array, size=(resize, resize))
        data_channel_2 = scipy.misc.imresize(arr = sum_array, size=(resize, resize))
        data_channel_3 = scipy.misc.imresize(arr = var_array, size=(resize, resize))

        # add all the channels to the channeled data
        for r in range(0,len(data_channel_1)):
            for c in range(0,len(data_channel_1[0])):
                Channeled_Data[r][c][0] = data_channel_1[r][c]
                Channeled_Data[r][c][1] = data_channel_2[r][c]
                Channeled_Data[r][c][2] = data_channel_3[r][c]

        if(index < training_length):
            X_train[index] = Channeled_Data
            Y_train[index] = int(is_threat)
        else:
            X_test[index - training_length] = Channeled_Data
            Y_test[index - training_length] = int(is_threat)

        return X_train, Y_train, X_test, Y_test

if __name__ == '__main__':

    # Example to fine-tune on 3000 samples from Cifar10

    img_rows, img_cols = 224, 224 # Resolution of inputs
    channel = 3
    num_classes = 2
    batch_size = 16 
    nb_epoch = 10

    # Load training and eval data
    """sc = SupervisedClassifier('../../stage1_labels.csv')
    dataCreator = TensorFlowDataSetCreator(sc)
    print('RUNNING ON ALL SEGMENTS, WITH AUGMENTATION')
    #dataset = dataCreator.CreateTensorFlowDataSetFromBlockStream(channels = channel, block_size = 56, resize = -1, segmentNumber = -100, image_filepath = "../../../rec/data/PSRC/Data/stage1/a3d/", nii_filepath = "../visualize/data/Batch_2D_warp_labels/")
    dataset = dataCreator.CreateTensorFlowDataSetFromBlockStream(channels = channel, block_size = 56, resize = -1, image_number = 2, segmentNumber = -100, image_filepath = "../visualize/data/a3d/", nii_filepath = "../visualize/data/Batch_2D_warp_labels/") 
    X_train = dataset.getTrainingData()
    list_of_values_train = dataset.getTrainingLabels()
    Y_train = to_categorical(list_of_values_train, num_classes=2)
    X_valid = dataset.getTestingData()
    list_of_values_test = dataset.getTestingLabels()
    Y_valid = to_categorical(list_of_values_test, num_classes=2)"""

    filepath = "generated_blocks/block_size_56/"

    max_folder = filepath+"max/"
    sum_folder = filepath+"sum/"
    var_folder = filepath+"var/"

    X_train, Y_train, X_valid, Y_valid = load_images_from_folder(max_folder, sum_folder, var_folder, img_rows)

    print len(X_train)
    print len(Y_train)
    print len(X_valid)
    print len(Y_valid)
    
    # Load our model
    model = resnet50_model(img_rows, img_cols, channel, num_classes)

    # Start Fine-tuning
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              shuffle=True,
              verbose=1,
              validation_data=(X_valid, Y_valid),
              )

    # Make predictions
    predictions_valid = model.predict(X_valid, batch_size=batch_size, verbose=2)

    # Cross-entropy loss score
    score = log_loss(Y_valid, predictions_valid)
