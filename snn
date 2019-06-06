import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
# import scipy
from PIL import Image
# from scipy import ndimage
from utils import *

# %matplotlib inline
plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# %load_ext autoreload
# %autoreload 2

# np.random.seed(1)

train_name = 'catvnoncat'

# load dataset
train_x_orig, train_y, test_x_orig, test_y, classes = load_data(train_name)

# # Example of a picture
# index = 92
# plt.imshow(train_x_orig[index])
# plt.show()
# print ("y = " + str(train_y[0,index]) + ". It's a " + str(classes[train_y[0,index]]) +  " picture.")


# Explore your dataset 
# m_train = train_x_orig.shape[0]
# num_px = train_x_orig.shape[1]
# m_test = test_x_orig.shape[0]

# print ("Number of training examples: " + str(m_train))
# print ("Number of testing examples: " + str(m_test))
# print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
# print ("train_x_orig shape: " + str(train_x_orig.shape))
# print ("train_y shape: " + str(train_y.shape))
# print ("test_x_orig shape: " + str(test_x_orig.shape))
# print ("test_y shape: " + str(test_y.shape))


# Reshape the training and test examples 
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten[:,:] / 255.
test_x = test_x_flatten[:,:] / 255.

# train_x = train_x[:,:192]
# train_y = train_y[:,:192]

print ("train_x's shape: " + str(train_x.shape))
print ("test_x's shape: " + str(test_x.shape))

start = time.perf_counter()
layers_dims = [train_x.shape[0], 30, 12, 5, 1] #  4-layer model
optimization = {"beta1": 0.9, "beta2": 0.99,  "epsilon": 1e-8}
parameters = L_layer_model(train_x, train_y, layers_dims, learning_rate = 0.0030, num_iterations = 80, print_cost = True, regularization_lambd = 0.8, optimization = optimization, mini_batch_size = 64)
print("%.5fs"%(time.perf_counter() - start))
# 15s 0.94 0.74
# layers_dims = [train_x.shape[0], 20, 7, 5, 1] #  4-layer model
# parameters = L_layer_model(train_x, train_y, layers_dims, learning_rate = 0.0070, num_iterations = 1, print_cost = True, regularization_lambd = 0.75, optimization = optimization, mini_batch_size = 1)
#30 0.99 0.8
# layers_dims = [train_x.shape[0], 30, 12, 5, 1] #  4-layer model
# optimization = {"beta1": 0.9, "beta2": 0.99,  "epsilon": 1e-8}
# parameters = L_layer_model(train_x, train_y, layers_dims, learning_rate = 0.0030, num_iterations = 80, print_cost = True, regularization_lambd = 0.8, optimization = optimization, mini_batch_size = 64)

# save(train_name, parameters)

# parameters = load(train_name)

# predict_minsort(classes, train_x, train_y, parameters, (64,64,3))
pred_train = predict(train_x, train_y, parameters)
pred_test = predict(test_x, test_y, parameters)

# print_mislabeled_images(classes, test_x, test_y, pred_test, (64,64,3))


# my_image = "my_image.jpg" # change this to the name of your image file 
# my_label_y = [1] # the true class of your image (1 -> cat, 0 -> non-cat)

# fname = "images/" + my_image
# image = np.array(ndimage.imread(fname, flatten=False))
# my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((num_px*num_px*3,1))
# my_image = my_image/255.
# my_predicted_image = predict(my_image, my_label_y, parameters)

# plt.imshow(image)
# print ("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
