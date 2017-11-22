import keras
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K

# Set values
n = 224
batch_size = 32
nb_epoch = 20
nb_phase_two_epoch = 20
nb_classes = 2
heavy_augmentation = False

# Load Data
X_train = np.load("../cnn_finetune/X_train_56_blocksize_3_channel_size224.npy")
print('loaded')
Y_train = np.load("../cnn_finetune/Y_train_56_blocksize_3_channel_size224.npy")
print('loaded')
X_valid = np.load("../cnn_finetune/X_valid_56_blocksize_3_channel_size224.npy")
print('loaded')
Y_valid = np.load("../cnn_finetune/Y_valid_56_blocksize_3_channel_size224.npy")
print('loaded')

# Generate dummy data
"""X_train = np.random.random((1000, n, n, 3))
Y_train = keras.utils.to_categorical(np.random.randint(2, size=(1000, 1)), num_classes=2)
X_valid = np.random.random((200, n, n, 3))
Y_valid = keras.utils.to_categorical(np.random.randint(2, size=(200, 1)), num_classes=2)"""

if heavy_augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=45,
        width_shift_range=0.25,
        height_shift_range=0.25,
        horizontal_flip=True,
        vertical_flip=False,
        zoom_range=0.5,
        channel_shift_range=0.5,
        fill_mode='nearest')
else:
    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=0,
        width_shift_range=0.125,
        height_shift_range=0.125,
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode='nearest')

datagen.fit(X_train)

# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)

# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)

# and a logistic layer -- let's say we have 200 classes
predictions = Dense(nb_classes, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# train the model on the new data for a few epochs
model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size, shuffle=True),
            samples_per_epoch=X_train.shape[0],
            nb_epoch=nb_epoch,
            validation_data=datagen.flow(X_valid, Y_valid, batch_size=batch_size),
            nb_val_samples=X_valid.shape[0],
            )
# fit_generator(self, generator, steps_per_epoch, epochs=1, verbose=1, callbacks=None, validation_data=None, validation_steps=None, class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False, shuffle=True, initial_epoch=0)

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:249]:
   layer.trainable = False
for layer in model.layers[249:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers

print "fine-tuning top 2 inception blocks alongside the top dense layers"

for i in range(1,11):
    print "mega-epoch %d/10" % i
    model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size, shuffle=True),
            samples_per_epoch=X_train.shape[0],
            nb_epoch=nb_phase_two_epoch,
            validation_data=datagen.flow(X_valid, Y_valid, batch_size=batch_size),
            nb_val_samples=X_test.shape[0],
            )