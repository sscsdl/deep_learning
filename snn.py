import time
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
# import scipy
from PIL import Image
# from scipy import ndimage
from snn_utils import *

# %matplotlib inline
plt.rcParams['figure.figsize'] = (5.0, 5.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# %load_ext autoreload
# %autoreload 2

# np.random.seed(1)

# train_name = 'signs'
train_name = 'catvnoncat'

# load dataset
train_x_orig, train_y_orig, test_x_orig, test_y_orig, classes = load_data('catvnoncat')
# train_x_orig, train_y_orig, test_x_orig, test_y_orig, classes = load_data('signs')
# train_x_orig, train_y_orig, test_x_orig, test_y_orig, classes = load_data('like')

# Explore your dataset 
# m_train = train_x_orig.shape[0]
# num_px = train_x_orig.shape[1]
# m_test = test_x_orig.shape[0]

# print ("Number of training examples: " + str(m_train))
# print ("Number of testing examples: " + str(m_test))
# print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
# print ("train_x_orig shape: " + str(train_x_orig.shape))
# print ("train_y_orig shape: " + str(train_y_orig.shape))
# print ("test_x_orig shape: " + str(test_x_orig.shape))
# print ("test_y shape: " + str(test_y.shape))


# Reshape the training and test examples 
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten[:,:] / 255.
test_x = test_x_flatten[:,:] / 255.

# 多分类任务one hot处理
max_num = np.squeeze(np.max(train_y_orig, axis=1))
if max_num > 1:
    train_y = convert_to_one_hot(train_y_orig, max_num + 1)
    test_y = convert_to_one_hot(test_y_orig, max_num + 1)
else:
    train_y = train_y_orig
    test_y = test_y_orig


print ("train_x's shape: " + str(train_x.shape))
print ("test_x's shape: " + str(test_x.shape))
print ("train_y's shape: " + str(train_y.shape))
print ("test_y's shape: " + str(test_y.shape))
if max_num == 1:
    print("train_y's true: " + str(np.sum(train_y)) + ' %.1f%%'%(np.sum(train_y) / train_y.shape[1] * 100))
    print("test_y's true: " + str(np.sum(test_y)) + ' %.1f%%'%(np.sum(test_y) / test_y.shape[1] * 100))
# print(test_y)
# exit()

start = time.perf_counter()

if max_num > 1:
    last_layer = max_num + 1
else:
    last_layer = 1
layers_dims = [train_x.shape[0], 25, 12, last_layer] #  4-layer model
optimization = {"beta1": 0.9, "beta2": 0.999,  "epsilon": 1e-8}
# optimization = None
learning_rate = 0.0001
learning_decay_rate = 1
regularization_lambd = 0
num_iterations = 30
mini_batch_size = 32
parameters, costs = L_layer_model(train_x, train_y, layers_dims, learning_rate = learning_rate, learning_decay_rate = learning_decay_rate, num_iterations = num_iterations, print_cost = True, regularization_lambd = regularization_lambd, optimization = optimization, mini_batch_size = mini_batch_size, Y_orig = train_y_orig)

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
pred_train, tran_accuracy = predict(train_x, train_y_orig, parameters)
pred_test, test_accuracy = predict(test_x, test_y_orig, parameters)

# plot the cost
# plt.plot(np.squeeze(costs))
# plt.ylabel('cost')
# plt.xlabel('iterations (per tens)')
# plt.title("Learning rate = %.5f"%learning_rate + 
#             ' decay_rate = %.4f'%learning_decay_rate + 
#             "\n" + 
#             'regularization = ' + str(regularization_lambd) + 
#             ' optimization = ' + str(0 if optimization==None else 1) + 
#             "\n" + 
#             "Tran Accuracy = %.3f "%tran_accuracy +
#             "Test Accuracy = %.3f"%test_accuracy
#         )
# plt.savefig('D:\www\snn\plot\%.5f-%.4f-%.3f-%.3f.png'%(learning_rate,learning_decay_rate,tran_accuracy,test_accuracy))
# plt.savefig('D:\www\snn\plot\%s.png'%time.strftime("%Y-%m-%d %H-%M-%S", time.localtime()))
# plt.show()
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