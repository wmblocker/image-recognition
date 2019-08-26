import matplotlib.pyplot as plt
import numpy as np

# Ciphar 10 Dataset for input
from keras.datasets import cifar10

# Linear Stack of Neural Network Layers
from keras.models import Sequential

# Core Layers that will help train our model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D

# Utilities to help transform data
from keras.utils import np_utils


# Set the pseudo random number generator
np.random.seed(123)


# Load dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data();

print ("Before setting Shape of Dataset")
print (x_train.shape)


# Ensure that the the dataset has a shape of 32x32 and a depth of 3 RGB Channels
x_train = x_train.reshape(x_train.shape[0], 3, 32, 32)
x_test = x_test.reshape(x_test.shape[0], 3, 32, 32)

print ("After Setting the shape")
print (x_train.shape)

# Convert to Float32 and normalize data values to [0,1]
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train /= 255
x_test /= 255

# Split the test data into 10 distinct class labels
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

model = Sequential()

model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(3, 32, 32)))

# Prevents overfitting
model.add(Dropout(0.25))


# Weights from Convolution Layers Must be flattened before passing to Dense layer
model.add(Flatten())

model.add(Dense(128, activation='relu'))

# Prevents overfitting
model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))

print "Model Shape"
print model.output_shape

# 8. Compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# 9. Fit model on training data
model.fit(x_train, y_train,
          batch_size=32, nb_epoch=50, verbose=1)

# 10. Evaluate model on test data
score = model.evaluate(x_test, y_test, verbose=0)

