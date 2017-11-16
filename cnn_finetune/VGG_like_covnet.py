import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.metrics import confusion_matrix
from keras.optimizers import SGD

image_size = 58

"""# Generate dummy data
X_train = np.random.random((100, image_size, image_size, 3))
Y_train = keras.utils.to_categorical(np.random.randint(2, size=(100, 1)), num_classes=2)
X_valid = np.random.random((20, image_size, image_size, 3))
Y_valid = keras.utils.to_categorical(np.random.randint(2, size=(20, 1)), num_classes=2)"""

# Read in data
X_train = np.load("X_train_56_blocksize_3_channel_size58.npy")
print 'loaded'
Y_train = np.load("Y_train_56_blocksize_3_channel_size58.npy")
print 'loaded'
X_valid = np.load("X_valid_56_blocksize_3_channel_size58.npy")
print 'loaded'
Y_valid = np.load("Y_valid_56_blocksize_3_channel_size58.npy")
print 'loaded'

model = Sequential()
# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(image_size, image_size, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

#model.fit(x_train, y_train, batch_size=32, epochs=10)
#y_pred = model.predict(x_test)

#print y_pred
#print y_test

#score = model.evaluate(x_test, y_test, batch_size=32)
#print score

# Start Fine-tuning
model.fit(X_train, Y_train,
          batch_size=32,
          nb_epoch=10,
          shuffle=True,
          verbose=1,
          validation_data=(X_valid, Y_valid),
          )

# Make predictions
predictions_valid = model.predict(X_valid, batch_size=32, verbose=2)

print predictions_valid
print Y_valid

# Cross-entropy loss score
#score = log_loss(Y_valid, predictions_valid)
score = model.evaluate(X_valid, Y_valid, batch_size=32)
print score