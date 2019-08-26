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
x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)
x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)

print ("After Setting the shape")
print (x_train.shape)

plt.imshow(x_train[0])
plt.show()