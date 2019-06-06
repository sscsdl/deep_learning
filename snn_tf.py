import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from tf_utils import *

# %matplotlib inline
np.random.seed(1)

train_name = 'tfsigns'

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_data('signs')

# # Example of a picture
# index = 3
# plt.imshow(X_train_orig[index])
# plt.show()
# print ("y = " + str(np.squeeze(Y_train_orig[:, index])))


# Flatten the training and test images
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
# Normalize image vectors
X_train = X_train_flatten/255.
X_test = X_test_flatten/255.
# Convert training and test labels to one hot matrices
Y_train = convert_to_one_hot(Y_train_orig, 6)
Y_test = convert_to_one_hot(Y_test_orig, 6)

print ("number of training examples = " + str(X_train.shape[1]))
print ("number of test examples = " + str(X_test.shape[1]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))

parameters = model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001,
          num_epochs = 30, minibatch_size = 32, print_cost = True)
# print(parameters)

# save(train_name, parameters)

# parameters = load(train_name)

# import scipy
# from PIL import Image
# from scipy import ndimage

# ## START CODE HERE ## (PUT YOUR IMAGE NAME) 
# my_image = "thumbs_up.jpg"
# ## END CODE HERE ##

# # We preprocess your image to fit your algorithm.
# fname = "images/" + my_image
# image = np.array(ndimage.imread(fname, flatten=False))
# my_image = scipy.misc.imresize(image, size=(64,64)).reshape((1, 64*64*3)).T
# my_image_prediction = predict(my_image, parameters)

# plt.imshow(image)
# print("Your algorithm predicts: y = " + str(np.squeeze(my_image_prediction)))
