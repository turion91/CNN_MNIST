# -*- coding: utf-8 -*-
"""Contains the dense layers functions.
"""

__author__ = 'Adriano Vereno'

__status__ = 'Development'
__version__ = '0.0.1'
__date__ = '2018-02-06'

import tensorflow as tf
import variables as cnn_mnist

from numpy.random import seed
seed(42)
tf.set_random_seed(42)

def weight_variable(shape):
    '''
    Set a function to create the initial weights from a truncated normal distribution
    '''

    initial = tf.truncated_normal(shape, stddev=cnn_mnist.w_std, seed=42)
    return tf.Variable(initial)

def bias_variable(shape):
    '''
    Create the initial biases
    '''

    initial = tf.constant(cnn_mnist.b_cont, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    '''
    set the convolutionary layer with a stride of 1
    the output image will have the same size of the input one
    '''

    return tf.nn.conv2d(x, W, strides=cnn_mnist.conv_strides, padding='SAME')

def max_pool_2x2(x):
    '''
    set the maxpool layer with a filter size of 2*2 and a stride of 2
    the output image will have the same size of the input one
    '''
    if cnn_mnist.max_pooling == True:
        return tf.nn.max_pool(x, ksize=cnn_mnist.ksize,
                              strides=cnn_mnist.pool_strides, padding='SAME')
    else:
        return tf.nn.avg_pool(x, ksize=cnn_mnist.ksize,
                              strides=cnn_mnist.pool_strides, padding='SAME')



def dense1(nb_conv, flat_layer, nb_hidden_neurons_1, input_dim, keep_prob, drop_dense1=cnn_mnist.drop_dense1
           , nb_filters_1=cnn_mnist.nb_filters_1, nb_filters_2=cnn_mnist.nb_filters_2
           , nb_filters_3=cnn_mnist.nb_filters_3, nb_filters_4=cnn_mnist.nb_filters_4):

    '''
    Create a module with one dense and the final layer and returns the last layer.It is possible
    to detach dropout layer.
    nb_conv --> nb of convolution layers
    input_dim --> input dimension
    keep_prob --> a tensor for the dropout probability
    drop_dense1 --> boolean for whether or not we keep the dropout layer
    nb_filters_1 --> nb of filters
    '''
    #creates the weights placeholders based on input dimensions
    if nb_conv == 1:
        w1_dense = weight_variable([input_dim*nb_filters_1, nb_hidden_neurons_1])
        h_pool2_flat = tf.reshape(flat_layer, [-1, input_dim*nb_filters_1])
    elif nb_conv == 2:
        w1_dense = weight_variable([input_dim*nb_filters_2, nb_hidden_neurons_1])
        h_pool2_flat = tf.reshape(flat_layer, [-1, input_dim*nb_filters_2])
    elif nb_conv == 3:
        w1_dense = weight_variable([input_dim*nb_filters_3, nb_hidden_neurons_1])
        h_pool2_flat = tf.reshape(flat_layer, [-1, input_dim*nb_filters_3])
    else:
        w1_dense = weight_variable([input_dim*nb_filters_4, nb_hidden_neurons_1])
        h_pool2_flat = tf.reshape(flat_layer, [-1, input_dim*nb_filters_4])
    #creates bias placeholder for the first dense layer
    b1_dense = bias_variable([nb_hidden_neurons_1])

    #same but for the last layer
    w2_dense = weight_variable([nb_hidden_neurons_1, cnn_mnist.nb_labels])
    b2_dense = bias_variable([cnn_mnist.nb_labels])


    #creates the dense layer, which is basically a normal hidden layer that will take
    #the input of the previous pooling layer flattened.
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w1_dense) + b1_dense)

    ##########################

    #creates the placeholder and the
    #dropout layer
    if  drop_dense1 == True:
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    else:
        h_fc1_drop = h_fc1

    #creates the last dense layer
    y_conv = tf.matmul(h_fc1_drop, w2_dense) + b2_dense
    return y_conv

def dense2(nb_conv, flat_layer, nb_hidden_neurons_1, nb_hidden_neurons_2
           , input_dim, keep_prob, drop_dense1=cnn_mnist.drop_dense1
           , drop_dense2=cnn_mnist.drop_dense2
           , nb_filters_1=cnn_mnist.nb_filters_1, nb_filters_2=cnn_mnist.nb_filters_2
           , nb_filters_3=cnn_mnist.nb_filters_3, nb_filters_4=cnn_mnist.nb_filters_4):

    '''
    Create a module with two denses and the final layer and returns the last layer.It is possible
    to detach dropout layer.
    nb_conv --> nb of convolution layers
    input_dim --> input dimension
    keep_prob --> a tensor for the dropout probability
    drop_dense1 --> boolean for whether or not we keep the dropout layer
    nb_filters_1 --> nb of filters
    '''
    #creates the weights placeholders based on input dimensions
    if nb_conv == 1:
        w1_dense = weight_variable([input_dim*nb_filters_1, nb_hidden_neurons_1])
        h_pool2_flat = tf.reshape(flat_layer, [-1, input_dim*nb_filters_1])
    elif nb_conv == 2:
        w1_dense = weight_variable([input_dim*nb_filters_2, nb_hidden_neurons_1])
        h_pool2_flat = tf.reshape(flat_layer, [-1, input_dim*nb_filters_2])
    elif nb_conv == 3:
        w1_dense = weight_variable([input_dim*nb_filters_3, nb_hidden_neurons_1])
        h_pool2_flat = tf.reshape(flat_layer, [-1, input_dim*nb_filters_3])
    else:
        w1_dense = weight_variable([input_dim*nb_filters_4, nb_hidden_neurons_1])
        h_pool2_flat = tf.reshape(flat_layer, [-1, input_dim*nb_filters_4])
    #creates bias placeholder for the first dense layer
    b1_dense = bias_variable([nb_hidden_neurons_1])

    #same but for the second layer
    w2_dense = weight_variable([nb_hidden_neurons_1, nb_hidden_neurons_2])
    b2_dense = bias_variable([nb_hidden_neurons_2])

    #same but for the last layer
    w3_dense = weight_variable([nb_hidden_neurons_2, cnn_mnist.nb_labels])
    b3_dense = bias_variable([cnn_mnist.nb_labels])


    #creates the dense layer, which is basically a normal hidden layer that will take
    #the input of the previous pooling layer flattened.
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w1_dense) + b1_dense)

    ##########################

    #creates the placeholder and the
    #dropout layer
    if  drop_dense1 == True:
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    else:
        h_fc1_drop = h_fc1

    #creates the dense layer, which is basically a normal hidden layer that will take
    #the input of the previous pooling layer flattened.
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, w2_dense) + b2_dense)

    ##########################

    #creates the placeholder and the
    #dropout layer
    if  drop_dense2 == True:
        h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)
    else:
        h_fc2_drop = h_fc2

    #creates the last dense layer
    y_conv = tf.matmul(h_fc2_drop, w3_dense) + b3_dense
    return y_conv

def dense3(nb_conv, flat_layer, nb_hidden_neurons_1, nb_hidden_neurons_2, nb_hidden_neurons_3
           , input_dim, keep_prob, drop_dense1=cnn_mnist.drop_dense1
           , drop_dense2=cnn_mnist.drop_dense2, drop_dense3=cnn_mnist.drop_dense3
           , nb_filters_1=cnn_mnist.nb_filters_1, nb_filters_2=cnn_mnist.nb_filters_2
           , nb_filters_3=cnn_mnist.nb_filters_3, nb_filters_4=cnn_mnist.nb_filters_4):

    '''
    Create a module with three denses and the final layer and returns the last layer.It is possible
    to detach dropout layer.
    nb_conv --> nb of convolution layers
    input_dim --> input dimension
    keep_prob --> a tensor for the dropout probability
    drop_dense1 --> boolean for whether or not we keep the dropout layer
    nb_filters_1 --> nb of filters
    '''
    #creates the weights placeholders based on input dimensions
    if nb_conv == 1:
        w1_dense = weight_variable([input_dim*nb_filters_1, nb_hidden_neurons_1])
        h_pool2_flat = tf.reshape(flat_layer, [-1, input_dim*nb_filters_1])
    elif nb_conv == 2:
        w1_dense = weight_variable([input_dim*nb_filters_2, nb_hidden_neurons_1])
        h_pool2_flat = tf.reshape(flat_layer, [-1, input_dim*nb_filters_2])
    elif nb_conv == 3:
        w1_dense = weight_variable([input_dim*nb_filters_3, nb_hidden_neurons_1])
        h_pool2_flat = tf.reshape(flat_layer, [-1, input_dim*nb_filters_3])
    else:
        w1_dense = weight_variable([input_dim*nb_filters_4, nb_hidden_neurons_1])
        h_pool2_flat = tf.reshape(flat_layer, [-1, input_dim*nb_filters_4])
    #creates bias placeholder for the first dense layer
    b1_dense = bias_variable([nb_hidden_neurons_1])

    #same but for the second layer
    w2_dense = weight_variable([nb_hidden_neurons_1, nb_hidden_neurons_2])
    b2_dense = bias_variable([nb_hidden_neurons_2])

    #same but for the third layer
    w3_dense = weight_variable([nb_hidden_neurons_2, nb_hidden_neurons_3])
    b3_dense = bias_variable([nb_hidden_neurons_3])

    #same but for the last layer
    w4_dense = weight_variable([nb_hidden_neurons_3, cnn_mnist.nb_labels])
    b4_dense = bias_variable([cnn_mnist.nb_labels])


    #creates the dense layer, which is basically a normal hidden layer that will take
    #the input of the previous pooling layer flattened.
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w1_dense) + b1_dense)

    ##########################

    #creates the placeholder and the
    #dropout layer
    if  drop_dense1 == True:
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    else:
        h_fc1_drop = h_fc1

    #creates the dense layer, which is basically a normal hidden layer that will take
    #the input of the previous pooling layer flattened.
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, w2_dense) + b2_dense)

    ##########################

    #creates the placeholder and the
    #dropout layer
    if  drop_dense2 == True:
        h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)
    else:
        h_fc2_drop = h_fc2

    #creates the dense layer, which is basically a normal hidden layer that will take
    #the input of the previous pooling layer flattened.
    h_fc3 = tf.nn.relu(tf.matmul(h_fc2_drop, w3_dense) + b3_dense)

    ##########################

    #creates the placeholder and the
    #dropout layer
    if  drop_dense3 == True:
        h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob)
    else:
        h_fc3_drop = h_fc3

    #creates the last dense layer
    y_conv = tf.matmul(h_fc3_drop, w4_dense) + b4_dense
    return y_conv

def dense4(nb_conv, flat_layer, nb_hidden_neurons_1, nb_hidden_neurons_2
           , nb_hidden_neurons_3, nb_hidden_neurons_4
           , input_dim, keep_prob, drop_dense1=cnn_mnist.drop_dense1
           , drop_dense2=cnn_mnist.drop_dense2, drop_dense3=cnn_mnist.drop_dense3
           , drop_dense4=cnn_mnist.drop_dense4
           , nb_filters_1=cnn_mnist.nb_filters_1, nb_filters_2=cnn_mnist.nb_filters_2
           , nb_filters_3=cnn_mnist.nb_filters_3, nb_filters_4=cnn_mnist.nb_filters_4):

    '''
    Create a module with four denses and the final layer and returns the last layer.It is possible
    to detach dropout layer.
    nb_conv --> nb of convolution layers
    input_dim --> input dimension
    keep_prob --> a tensor for the dropout probability
    drop_dense1 --> boolean for whether or not we keep the dropout layer
    nb_filters_1 --> nb of filters
    '''
    #creates the weights placeholders based on input dimensions
    if nb_conv == 1:
        w1_dense = weight_variable([input_dim*nb_filters_1, nb_hidden_neurons_1])
        h_pool2_flat = tf.reshape(flat_layer, [-1, input_dim*nb_filters_1])
    elif nb_conv == 2:
        w1_dense = weight_variable([input_dim*nb_filters_2, nb_hidden_neurons_1])
        h_pool2_flat = tf.reshape(flat_layer, [-1, input_dim*nb_filters_2])
    elif nb_conv == 3:
        w1_dense = weight_variable([input_dim*nb_filters_3, nb_hidden_neurons_1])
        h_pool2_flat = tf.reshape(flat_layer, [-1, input_dim*nb_filters_3])
    else:
        w1_dense = weight_variable([input_dim*nb_filters_4, nb_hidden_neurons_1])
        h_pool2_flat = tf.reshape(flat_layer, [-1, input_dim*nb_filters_4])
    #creates bias placeholder for the first dense layer
    b1_dense = bias_variable([nb_hidden_neurons_1])

    #same but for the second layer
    w2_dense = weight_variable([nb_hidden_neurons_1, nb_hidden_neurons_2])
    b2_dense = bias_variable([nb_hidden_neurons_2])

    #same but for the third layer
    w3_dense = weight_variable([nb_hidden_neurons_2, nb_hidden_neurons_3])
    b3_dense = bias_variable([nb_hidden_neurons_3])

    #same but for the fourth layer
    w4_dense = weight_variable([nb_hidden_neurons_3, nb_hidden_neurons_4])
    b4_dense = bias_variable([nb_hidden_neurons_4])

    #same but for the last layer
    w5_dense = weight_variable([nb_hidden_neurons_4, cnn_mnist.nb_labels])
    b5_dense = bias_variable([cnn_mnist.nb_labels])


    #creates the dense layer, which is basically a normal hidden layer that will take
    #the input of the previous pooling layer flattened.
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w1_dense) + b1_dense)

    ##########################

    #creates the placeholder and the
    #dropout layer
    if  drop_dense1 == True:
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    else:
        h_fc1_drop = h_fc1

    #creates the dense layer, which is basically a normal hidden layer that will take
    #the input of the previous pooling layer flattened.
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, w2_dense) + b2_dense)

    ##########################

    #creates the placeholder and the
    #dropout layer
    if  drop_dense2 == True:
        h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)
    else:
        h_fc2_drop = h_fc2

    #creates the dense layer, which is basically a normal hidden layer that will take
    #the input of the previous pooling layer flattened.
    h_fc3 = tf.nn.relu(tf.matmul(h_fc2_drop, w3_dense) + b3_dense)

    ##########################

    #creates the placeholder and the
    #dropout layer
    if  drop_dense3 == True:
        h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob)
    else:
        h_fc3_drop = h_fc3

    #creates the dense layer, which is basically a normal hidden layer that will take
    #the input of the previous pooling layer flattened.
    h_fc4 = tf.nn.relu(tf.matmul(h_fc3_drop, w4_dense) + b4_dense)

    ##########################

    #creates the placeholder and the
    #dropout layer
    if  drop_dense4 == True:
        h_fc4_drop = tf.nn.dropout(h_fc4, keep_prob)
    else:
        h_fc4_drop = h_fc4
    #creates the last dense layer
    y_conv = tf.matmul(h_fc4_drop, w5_dense) + b5_dense
    return y_conv
