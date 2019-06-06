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

print ("train_x's shape: " + str(train_x.shape))
print ("test_x's shape: " + str(test_x.shape))

start = time.perf_counter()
# layers_dims = [train_x.shape[0], 20, 7, 5, 1] #  4-layer model
# optimization = {"beta1": 0.9, "beta2": 0.9,  "epsilon": 1e-8}
# parameters = L_layer_model(train_x, train_y, layers_dims, learning_rate = 0.0030, num_iterations = 2, print_cost = True, regularization_lambd = 0.70, optimization = optimization, mini_batch_size = 1)
layers_dims = [train_x.shape[0], 30, 12, 5, 1] #  4-layer model
optimization = {"beta1": 0.9, "beta2": 0.99,  "epsilon": 1e-8}
learning_rate = 0.0030
num_iterations = 80
print_cost = True
regularization_lambd = 0.8
optimization = optimization
mini_batch_size = 64

print({'layers_dims': layers_dims, 'learning_rate': learning_rate, 'num_iterations': num_iterations, 'print_cost': print_cost, 'regularization_lambd': regularization_lambd, 'optimization': optimization, 'mini_batch_size': mini_batch_size})

X, Y = train_x, train_y

costs = []                         # keep track of cost
t = 0

# Parameters initialization. (\u2248 1 line of code)
np.random.seed(1)
parameters = {}
L = len(layers_dims)            # number of layers in the network
for l in range(1, L):
    parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) / np.sqrt(layers_dims[l-1]) #*0.01
    parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

# Initialize the optimizer
if optimization:
    v, s = initialize_adam(parameters)

# Loop (gradient descent)
for i in range(0, num_iterations):

    minibatches = random_mini_batches(X, Y, mini_batch_size)

    k = 0
    for minibatch in minibatches:

        # Select a minibatch
        (minibatch_X, minibatch_Y) = minibatch

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        ### START CODE HERE ### (\u2248 1 line of code)
        # AL, caches = L_model_forward(X, parameters)
        caches = []
        A = X
        L = len(parameters) // 2                  # number of layers in the neural network
        
        # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
        for l in range(1, L):
            A_prev = A 
            # A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation = "relu")
            Z, linear_cache = linear_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)])
            A, activation_cache = relu(Z)
            cache = (linear_cache, activation_cache)
            caches.append(cache)
        
        # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
        # AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation = "sigmoid")
        Z, linear_cache = linear_forward(A, parameters['W' + str(L)], parameters['b' + str(L)])
        A, activation_cache = sigmoid(Z)
        cache = (linear_cache, activation_cache)
        AL = A
        caches.append(cache)
        ### END CODE HERE ###
        
        # Compute cost.
        ### START CODE HERE ### (\u2248 1 line of code)
        if regularization_lambd == 0:
            # cost = compute_cost(AL, Y)
            m = Y.shape[1]
            # Compute loss from aL and y.
            cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
            cost = np.squeeze(cost)
        else:
            # cost = compute_cost_with_regularization(AL, Y, parameters, regularization_lambd)
            L = len(parameters) // 2
            m = Y.shape[1]
            cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
            cross_entropy_cost = np.squeeze(cost)
            parameters_sum = 0
            for l in range(1, L):
                parameters_sum += np.sum(np.square(parameters['W' + str(l)]))
            L2_regularization_cost = regularization_lambd * parameters_sum / (2 * m)
            ### END CODER HERE ###
            
            cost = cross_entropy_cost + L2_regularization_cost
        ### END CODE HERE ###
    
        # Backward propagation.
        ### START CODE HERE ### (\u2248 1 line of code)
        if regularization_lambd == 0:
            # grads = L_model_backward(AL, Y, caches)
            grads = {}
            L = len(caches) # the number of layers
            m = AL.shape[1]
            Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
            
            # Initializing the backpropagation
            dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
            
            # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
            current_cache = caches[L-1]
            # grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation = "sigmoid")
            linear_cache, activation_cache = current_cache
            dZ = sigmoid_backward(dAL, activation_cache)
            grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_backward(dZ, linear_cache, regularization_lambd)
            
            for l in reversed(range(L-1)):
                # lth layer: (RELU -> LINEAR) gradients.
                current_cache = caches[l]
                # dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, activation = "relu")
                linear_cache, activation_cache = current_cache
                dZ = relu_backward(grads["dA" + str(l + 1)], activation_cache)
                dA_prev_temp, dW_temp, db_temp = linear_backward(dZ, linear_cache, regularization_lambd)
                grads["dA" + str(l)] = dA_prev_temp
                grads["dW" + str(l + 1)] = dW_temp
                grads["db" + str(l + 1)] = db_temp
        else:
            # grads = L_model_backward_with_regularization(AL, Y, caches, regularization_lambd)
            grads = {}
            L = len(caches) # the number of layers
            m = AL.shape[1]
            Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
            
            # Initializing the backpropagation
            dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
            
            # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
            current_cache = caches[L-1]
            # grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation = "sigmoid", regularization_lambd = regularization_lambd)
            linear_cache, activation_cache = current_cache
            dZ = sigmoid_backward(dAL, activation_cache)
            grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_backward(dZ, linear_cache, regularization_lambd)

            for l in reversed(range(L-1)):
                # lth layer: (RELU -> LINEAR) gradients.
                current_cache = caches[l]
                # dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, activation = "relu", regularization_lambd = regularization_lambd)
                linear_cache, activation_cache = current_cache
                dZ = relu_backward(grads["dA" + str(l + 1)], activation_cache)
                dA_prev_temp, dW_temp, db_temp = linear_backward(dZ, linear_cache, regularization_lambd)
                grads["dA" + str(l)] = dA_prev_temp
                grads["dW" + str(l + 1)] = dW_temp
                grads["db" + str(l + 1)] = db_temp
        ### END CODE HERE ###

        # Update parameters.
        ### START CODE HERE ### (\u2248 1 line of code)
        if optimization:
            t = t + 1 # Adam counter
            parameters, v, s = update_parameters_with_adam(parameters, grads, v, s, t, learning_rate, optimization['beta1'], optimization['beta2'], optimization['epsilon'])
        else:
            parameters = update_parameters(parameters, grads, learning_rate)
        ### END CODE HERE ###
                
    # Print the cost every 100 training example
    if print_cost and i % 10 == 0:
        print ("Cost after iteration %i: %f" %(i, cost))
    if print_cost and i % 10 == 0:
        costs.append(cost)
    k += 1
        
if print_cost:
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()


# 15s 0.94 0.74
# layers_dims = [train_x.shape[0], 20, 7, 5, 1] #  4-layer model
# parameters = L_layer_model(train_x, train_y, layers_dims, learning_rate = 0.0070, num_iterations = 1, print_cost = True, regularization_lambd = 0.75, optimization = optimization, mini_batch_size = 1)

# save(train_name, parameters)


# save('h5test', {'w':[1,2,3,4]})
# a = load('h5test')
# print(a)

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
