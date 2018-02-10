# -*- coding: utf-8 -*-
"""Contains the conv/pool layers functions.
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
        


def cp1(data, keep_prob, pool_conv1=cnn_mnist.pool_conv1,
        drop_conv1=cnn_mnist.drop_conv1, filter_size=cnn_mnist.filter_size
        , nb_filters_1=cnn_mnist.nb_filters_1):
    '''
    Create a module with one conv/pool and returns the final reshape layer.It is possible
    to detach dropout and pooling layer.
    data --> the input data
    keep_prob --> a tensor for the dropout probability
    pool_conv1 --> boolean for whether or not we keep the pooling layer
    drop_conv1 --> boolean for whether or not we keep the dropout layer
    filter_size --> the fiter size
    nb_filters_1 --> nb of filters
    '''
    w1_conv = weight_variable([filter_size, filter_size
                               , cnn_mnist.nb_channels, nb_filters_1])#5*5*1*32
    b1_conv = tf.Variable(tf.zeros([nb_filters_1]))#32

    #combinations to get the correct input dimensions for further computations
    if pool_conv1 == False:
        input_dim = cnn_mnist.input_dim_0_pool#28*28
    else:
        input_dim = cnn_mnist.input_dim_1_pool#14*14

    #presence/absence of pooling and dropout
    layer1_conv = conv2d(data, w1_conv)
    layer1_actv = tf.nn.relu(layer1_conv + b1_conv)
    if pool_conv1 == True:
        layer1_pool = max_pool_2x2(layer1_actv)
    else:
        layer1_pool = layer1_actv

    if  drop_conv1 == True:
        layer1_pool = tf.nn.dropout(layer1_pool, keep_prob)
    else:
        layer1_pool = layer1_pool

    #combinations of presence/absence of pooling layers
    if pool_conv1 == False:
        flat_layer = tf.reshape(layer1_pool, [-1, (cnn_mnist.input_dim_0_pool)*nb_filters_1])
    else:
        flat_layer = tf.reshape(layer1_pool, [-1, input_dim*nb_filters_1])
    return flat_layer, input_dim

def cp2(data, keep_prob, pool_conv1=cnn_mnist.pool_conv1, pool_conv2=cnn_mnist.pool_conv2
        , drop_conv1=cnn_mnist.drop_conv1, drop_conv2=cnn_mnist.drop_conv2
        , filter_size=cnn_mnist.filter_size, nb_filters_1=cnn_mnist.nb_filters_1
        , nb_filters_2=cnn_mnist.nb_filters_2):

    '''
    Create a module with one conv/pool and returns the final reshape layer.It is possible
    to detach dropout and pooling layer.
    data --> the input data
    keep_prob --> a tensor for the dropout probability
    pool_conv1 --> boolean for whether or not we keep the pooling layer
    drop_conv1 --> boolean for whether or not we keep the dropout layer
    filter_size --> the fiter size
    nb_filters_1 --> nb of filters
    '''
    w1_conv = weight_variable([filter_size, filter_size
                               , cnn_mnist.nb_channels, nb_filters_1])#5*5*1*32
    b1_conv = bias_variable([nb_filters_1])#32

    w2_conv = weight_variable([filter_size, filter_size, nb_filters_1, nb_filters_2])#5*5*32*64
    b2_conv = bias_variable([nb_filters_2])#64

    #combinations to get the correct input dimensions for further computations
    if pool_conv1 == False and pool_conv2 == False:
        input_dim = cnn_mnist.input_dim_0_pool#28*28
    elif pool_conv1 == False or pool_conv2 == False:
        input_dim = cnn_mnist.input_dim_1_pool#14*14
    else:
        input_dim = cnn_mnist.input_dim_2_pool#7*7

    #presence/absence of pooling and dropout
    layer1_conv = conv2d(data, w1_conv)
    layer1_actv = tf.nn.relu(layer1_conv + b1_conv)
    if pool_conv1 == True:
        layer1_pool = max_pool_2x2(layer1_actv)
    else:
        layer1_pool = layer1_actv

    if  drop_conv1 == True:
        layer1_pool = tf.nn.dropout(layer1_pool, keep_prob)
    else:
        layer1_pool = layer1_pool
    layer2_conv = conv2d(layer1_pool, w2_conv)
    layer2_actv = tf.nn.relu(layer2_conv + b2_conv)
    if pool_conv2 == True:
        layer2_pool = max_pool_2x2(layer2_actv)
    else:
        layer2_pool = layer2_actv

    if  drop_conv2 == True:
        layer2_pool = tf.nn.dropout(layer2_pool, keep_prob)
    else:
        layer2_pool = layer2_pool

    #combinations of presence/absence of pooling layers
    if pool_conv1 == False and pool_conv2 == False:
        flat_layer = tf.reshape(layer2_pool
                                , [-1, (cnn_mnist.input_dim_0_pool)*nb_filters_2])#-1*28*28*64
    elif pool_conv1 == False or pool_conv2 == False:
        flat_layer = tf.reshape(layer2_pool
                                , [-1, (cnn_mnist.input_dim_1_pool)*nb_filters_2])#-1*14*14*64
    else:
        flat_layer = tf.reshape(layer2_pool
                                , [-1, cnn_mnist.input_dim_2_pool*nb_filters_2])#-1*7*7*64

    return flat_layer, input_dim

def cp3(data, keep_prob, pool_conv1=cnn_mnist.pool_conv1, pool_conv2=cnn_mnist.pool_conv2
        , drop_conv1=cnn_mnist.drop_conv1, drop_conv2=cnn_mnist.drop_conv2
        , pool_conv3=cnn_mnist.pool_conv3, drop_conv3=cnn_mnist.drop_conv3
        , filter_size=cnn_mnist.filter_size, nb_filters_1=cnn_mnist.nb_filters_1
        , nb_filters_2=cnn_mnist.nb_filters_2, nb_filters_3=cnn_mnist.nb_filters_3):

    '''
    Create a module with one conv/pool and returns the final reshape layer.It is possible
    to detach dropout and pooling layer.
    data --> the input data
    keep_prob --> a tensor for the dropout probability
    pool_conv1 --> boolean for whether or not we keep the pooling layer
    drop_conv1 --> boolean for whether or not we keep the dropout layer
    filter_size --> the fiter size
    nb_filters_1 --> nb of filters
    '''

    w1_conv = weight_variable([filter_size, filter_size
                               , cnn_mnist.nb_channels, nb_filters_1])#5*5*1*32
    b1_conv = bias_variable([nb_filters_1])#32

    w2_conv = weight_variable([filter_size, filter_size
                               , nb_filters_1, nb_filters_2])#5*5*32*64
    b2_conv = bias_variable([nb_filters_2])#64

    w3_conv = weight_variable([filter_size, filter_size
                               , nb_filters_2, nb_filters_3])#5*5*64*128
    b3_conv = bias_variable([nb_filters_3])#128

    #combinations to get the correct input dimensions for further computations
    if pool_conv1 == False and pool_conv2 == False and pool_conv3 == False:
        input_dim = cnn_mnist.input_dim_0_pool#28*28

    elif pool_conv1 == False and pool_conv2 == False and pool_conv3 == True:
        input_dim = cnn_mnist.input_dim_1_pool#14*14
    elif pool_conv1 == False and pool_conv2 == True and pool_conv3 == False:
        input_dim = cnn_mnist.input_dim_1_pool#14*14
    elif pool_conv1 == True and pool_conv2 == False and pool_conv3 == False:
        input_dim = cnn_mnist.input_dim_1_pool##14*14


    elif pool_conv1 == False and pool_conv2 == True and pool_conv3 == True:
        input_dim = cnn_mnist.input_dim_2_pool#7*7
    elif pool_conv1 == True and pool_conv2 == False and pool_conv3 == True:
        input_dim = cnn_mnist.input_dim_2_pool#7*7
    elif pool_conv1 == True and pool_conv2 == True and pool_conv3 == False:
        input_dim = cnn_mnist.input_dim_2_pool#7*7
    elif pool_conv1 == False or pool_conv2 == False or pool_conv3 == False:
        input_dim = cnn_mnist.input_dim_1_pool#14*14

    else:
        input_dim = cnn_mnist.input_dim_3_pool#4*4

    #presence/absence of pooling and dropout
    layer1_conv = conv2d(data, w1_conv)
    layer1_actv = tf.nn.relu(layer1_conv + b1_conv)
    if pool_conv1 == True:
        layer1_pool = max_pool_2x2(layer1_actv)
    else:
        layer1_pool = layer1_actv

    if  drop_conv1 == True:
        layer1_pool = tf.nn.dropout(layer1_pool, keep_prob)
    else:
        layer1_pool = layer1_pool
    layer2_conv = conv2d(layer1_pool, w2_conv)
    layer2_actv = tf.nn.relu(layer2_conv + b2_conv)
    if pool_conv2 == True:
        layer2_pool = max_pool_2x2(layer2_actv)
    else:
        layer2_pool = layer2_actv

    if  drop_conv2 == True:
        layer2_pool = tf.nn.dropout(layer2_pool, keep_prob)
    else:
        layer2_pool = layer2_pool

    layer3_conv = conv2d(layer2_pool, w3_conv)
    layer3_actv = tf.nn.relu(layer3_conv + b3_conv)
    if pool_conv3 == True:
        layer3_pool = max_pool_2x2(layer3_actv)
    else:
        layer3_pool = layer3_actv

    if  drop_conv3 == True:
        layer3_pool = tf.nn.dropout(layer3_pool, keep_prob)
    else:
        layer3_pool = layer3_pool


    #combinations of presence/absence of pooling layers
    if pool_conv1 == False and pool_conv2 == False and pool_conv3 == False:
        flat_layer = tf.reshape(layer3_pool
                                , [-1, (cnn_mnist.input_dim_0_pool)*nb_filters_3])#-1*28*28*128
    elif pool_conv1 == False and pool_conv2 == False and pool_conv3 == True:
        flat_layer = tf.reshape(layer3_pool
                                , [-1, (cnn_mnist.input_dim_1_pool)*nb_filters_3])#-1*14*14*128
    elif pool_conv1 == False and pool_conv2 == True and pool_conv3 == False:
        flat_layer = tf.reshape(layer3_pool
                                , [-1, (cnn_mnist.input_dim_1_pool)*nb_filters_3])#-1*14*14*128
    elif pool_conv1 == True and pool_conv2 == False and pool_conv3 == False:
        flat_layer = tf.reshape(layer3_pool
                                , [-1, (cnn_mnist.input_dim_1_pool)*nb_filters_3])#-1*14*14*128

    elif pool_conv1 == False and pool_conv2 == True and pool_conv3 == True:
        flat_layer = tf.reshape(layer3_pool
                                , [-1, (cnn_mnist.input_dim_2_pool)*nb_filters_3])#-1*7*7*128
    elif pool_conv1 == True and pool_conv2 == False and pool_conv3 == True:
        flat_layer = tf.reshape(layer3_pool
                                , [-1, (cnn_mnist.input_dim_2_pool)*nb_filters_3])#-1*7*7*128
    elif pool_conv1 == True and pool_conv2 == True and pool_conv3 == False:
        flat_layer = tf.reshape(layer3_pool
                                , [-1, (cnn_mnist.input_dim_2_pool)*nb_filters_3])#-1*7*7*128
    elif pool_conv1 == False or pool_conv2 == False or pool_conv3 == False:
        flat_layer = tf.reshape(layer3_pool
                                , [-1, (cnn_mnist.input_dim_1_pool)*nb_filters_3])#-1*14*14*128

    else:
        flat_layer = tf.reshape(layer3_pool
                                , [-1, cnn_mnist.input_dim_3_pool*nb_filters_3]) #-1*4*4*128

    return flat_layer, input_dim

def cp4(data, keep_prob, pool_conv1=cnn_mnist.pool_conv1, pool_conv2=cnn_mnist.pool_conv2
        , drop_conv1=cnn_mnist.drop_conv1, drop_conv2=cnn_mnist.drop_conv2
        , pool_conv3=cnn_mnist.pool_conv3
        , drop_conv3=cnn_mnist.drop_conv3, pool_conv4=cnn_mnist.pool_conv4
        , drop_conv4=cnn_mnist.drop_conv4
        , filter_size=cnn_mnist.filter_size, nb_filters_1=cnn_mnist.nb_filters_1
        , nb_filters_2=cnn_mnist.nb_filters_2, nb_filters_3=cnn_mnist.nb_filters_3
        , nb_filters_4=cnn_mnist.nb_filters_4):
    '''
    Create a module with one conv/pool and returns the final reshape layer.It is possible
    to detach dropout and pooling layer.
    data --> the input data
    keep_prob --> a tensor for the dropout probability
    pool_conv1 --> boolean for whether or not we keep the pooling layer
    drop_conv1 --> boolean for whether or not we keep the dropout layer
    filter_size --> the fiter size
    nb_filters_1 --> nb of filters
    '''

    w1_conv = weight_variable([filter_size, filter_size
                               , cnn_mnist.nb_channels, nb_filters_1])#5*5*1*32
    b1_conv = bias_variable([nb_filters_1])#32

    w2_conv = weight_variable([filter_size, filter_size
                               , nb_filters_1, nb_filters_2])#5*5*32*64
    b2_conv = bias_variable([nb_filters_2])#64

    w3_conv = weight_variable([filter_size, filter_size
                               , nb_filters_2, nb_filters_3])#5*5*64*128
    b3_conv = bias_variable([nb_filters_3])#128

    w4_conv = weight_variable([filter_size, filter_size
                               , nb_filters_3, nb_filters_4])#5*5*128*256
    b4_conv = bias_variable([nb_filters_4])#256

    #combinations to get the correct input dimensions for further computations
    if pool_conv1 == False and pool_conv2 == False and pool_conv3 == False and pool_conv4 == False:
        input_dim = cnn_mnist.input_dim_0_pool


    elif pool_conv1 == False and pool_conv2 == False and pool_conv3 == False and pool_conv4 == True:
        input_dim = cnn_mnist.input_dim_1_pool
    elif pool_conv1 == False and pool_conv2 == False and pool_conv3 == True  and pool_conv4 == False:
        input_dim = cnn_mnist.input_dim_1_pool
    elif pool_conv1 == False and pool_conv2 == True and pool_conv3 == False  and pool_conv4 == False:
        input_dim = cnn_mnist.input_dim_1_pool
    elif pool_conv1 == True and pool_conv2 == False and pool_conv3 == False and pool_conv4 == False:
        input_dim = cnn_mnist.input_dim_1_pool


    elif pool_conv1 == False and pool_conv2 == False and pool_conv3 == True and pool_conv4 == True:
        input_dim = cnn_mnist.input_dim_2_pool
    elif pool_conv1 == False and pool_conv2 == True and pool_conv3 == False and pool_conv4 == True:
        input_dim = cnn_mnist.input_dim_2_pool
    elif pool_conv1 == True and pool_conv2 == False and pool_conv3 == False and pool_conv4 == True:
        input_dim = cnn_mnist.input_dim_2_pool

    elif pool_conv1 == True and pool_conv2 == False and pool_conv3 == True and pool_conv4 == False:
        input_dim = cnn_mnist.input_dim_2_pool
    elif pool_conv1 == False and pool_conv2 == True and pool_conv3 == True and pool_conv4 == False:
        input_dim = cnn_mnist.input_dim_2_pool


    elif pool_conv1 == True and pool_conv2 == True and pool_conv3 == False and pool_conv4 == False:
        input_dim = cnn_mnist.input_dim_2_pool


    elif pool_conv1 == True and pool_conv2 == True and pool_conv3 == True and pool_conv4 == False:
        input_dim = cnn_mnist.input_dim_3_pool
    elif pool_conv1 == False and pool_conv2 == True and pool_conv3 == True and pool_conv4 == True:
        input_dim = cnn_mnist.input_dim_3_pool
    elif pool_conv1 == True and pool_conv2 == False and pool_conv3 == True and pool_conv4 == True:
        input_dim = cnn_mnist.input_dim_3_pool
    elif pool_conv1 == True and pool_conv2 == True and pool_conv3 == False and pool_conv4 == True:
        input_dim = cnn_mnist.input_dim_3_pool
    elif pool_conv1 == False or pool_conv2 == False or pool_conv3 == False or pool_conv4 == False:
        input_dim = cnn_mnist.input_dim_1_pool
    else:
        input_dim = cnn_mnist.input_dim_4_pool

    #presence/absence of pooling and dropout
    layer1_conv = conv2d(data, w1_conv)
    layer1_actv = tf.nn.relu(layer1_conv + b1_conv)
    if pool_conv1 == True:
        layer1_pool = max_pool_2x2(layer1_actv)
    else:
        layer1_pool = layer1_actv

    if  drop_conv1 == True:
        layer1_pool = tf.nn.dropout(layer1_pool, keep_prob)
    else:
        layer1_pool = layer1_pool

    layer2_conv = conv2d(layer1_pool, w2_conv)
    layer2_actv = tf.nn.relu(layer2_conv + b2_conv)
    if pool_conv2 == True:
        layer2_pool = max_pool_2x2(layer2_actv)
    else:
        layer2_pool = layer2_actv

    if  drop_conv2 == True:
        layer2_pool = tf.nn.dropout(layer2_pool, keep_prob)
    else:
        layer2_pool = layer2_pool

    layer3_conv = conv2d(layer2_pool, w3_conv)
    layer3_actv = tf.nn.relu(layer3_conv + b3_conv)
    if pool_conv3 == True:
        layer3_pool = max_pool_2x2(layer3_actv)
    else:
        layer3_pool = layer3_actv

    if  drop_conv3 == True:
        layer3_pool = tf.nn.dropout(layer3_pool, keep_prob)
    else:
        layer3_pool = layer3_pool


    layer4_conv = conv2d(layer3_pool, w4_conv)
    layer4_actv = tf.nn.relu(layer4_conv + b4_conv)
    if pool_conv4 == True:
        layer4_pool = max_pool_2x2(layer4_actv)
    else:
        layer4_pool = layer4_actv

    if  drop_conv4 == True:
        layer4_pool = tf.nn.dropout(layer4_pool, keep_prob)
    else:
        layer4_pool = layer4_pool


    #combinations of presence/absence of pooling layers
    if pool_conv1 == False and pool_conv2 == False and pool_conv3 == False and pool_conv4 == False:
        flat_layer = tf.reshape(layer4_pool
                                , [-1, (cnn_mnist.input_dim_0_pool)*nb_filters_4])#-1*28*28*256

    elif pool_conv1 == False and pool_conv2 == False and pool_conv3 == False and pool_conv4 == True:
        flat_layer = tf.reshape(layer4_pool
                                , [-1, (cnn_mnist.input_dim_1_pool)*nb_filters_4])#-1*14*14*256
    elif pool_conv1 == False and pool_conv2 == False and pool_conv3 == True  and pool_conv4 == False:
        flat_layer = tf.reshape(layer4_pool
                                , [-1, (cnn_mnist.input_dim_1_pool)*nb_filters_4])#-1*14*14*256
    elif pool_conv1 == False and pool_conv2 == True and pool_conv3 == False  and pool_conv4 == False:
        flat_layer = tf.reshape(layer4_pool
                                , [-1, (cnn_mnist.input_dim_1_pool)*nb_filters_4])#-1*14*14*256
    elif pool_conv1 == True and pool_conv2 == False and pool_conv3 == False and pool_conv4 == False:
        flat_layer = tf.reshape(layer4_pool
                                , [-1, (cnn_mnist.input_dim_1_pool)*nb_filters_4])#-1*14*14*256


    elif pool_conv1 == False and pool_conv2 == False and pool_conv3 == True and pool_conv4 == True:
        flat_layer = tf.reshape(layer4_pool
                                , [-1, (cnn_mnist.input_dim_2_pool)*nb_filters_4])#-1*7*7*256
    elif pool_conv1 == False and pool_conv2 == True and pool_conv3 == False and pool_conv4 == True:
        flat_layer = tf.reshape(layer4_pool
                                , [-1, (cnn_mnist.input_dim_2_pool)*nb_filters_4])#-1*7*7*256
    elif pool_conv1 == True and pool_conv2 == False and pool_conv3 == False and pool_conv4 == True:
        flat_layer = tf.reshape(layer4_pool
                                , [-1, (cnn_mnist.input_dim_2_pool)*nb_filters_4])#-1*7*7*256

    elif pool_conv1 == True and pool_conv2 == False and pool_conv3 == True and pool_conv4 == False:
        flat_layer = tf.reshape(layer4_pool
                                , [-1, (cnn_mnist.input_dim_2_pool)*nb_filters_4])#-1*7*7*256
    elif pool_conv1 == False and pool_conv2 == True and pool_conv3 == True and pool_conv4 == False:
        flat_layer = tf.reshape(layer4_pool
                                , [-1, (cnn_mnist.input_dim_2_pool)*nb_filters_4])#-1*7*7*256


    elif pool_conv1 == True and pool_conv2 == True and pool_conv3 == False and pool_conv4 == False:
        flat_layer = tf.reshape(layer4_pool
                                , [-1, (cnn_mnist.input_dim_2_pool)*nb_filters_4])#-1*7*7*256


    elif pool_conv1 == True and pool_conv2 == True and pool_conv3 == True and pool_conv4 == False:
        flat_layer = tf.reshape(layer4_pool
                                , [-1, (cnn_mnist.input_dim_3_pool)*nb_filters_4])#-1*4*4*256
    elif pool_conv1 == False and pool_conv2 == True and pool_conv3 == True and pool_conv4 == True:
        flat_layer = tf.reshape(layer4_pool
                                , [-1, (cnn_mnist.input_dim_3_pool)*nb_filters_4])#-1*4*4*256
    elif pool_conv1 == True and pool_conv2 == False and pool_conv3 == True and pool_conv4 == True:
        flat_layer = tf.reshape(layer4_pool
                                , [-1, (cnn_mnist.input_dim_3_pool)*nb_filters_4])#-1*4*4*256
    elif pool_conv1 == True and pool_conv2 == True and pool_conv3 == False and pool_conv4 == True:
        flat_layer = tf.reshape(layer4_pool
                                , [-1, (cnn_mnist.input_dim_3_pool)*nb_filters_4])#-1*4*4*256

    elif pool_conv1 == False or pool_conv2 == False or pool_conv3 == False or pool_conv4 == False:
        flat_layer = tf.reshape(layer4_pool
                                , [-1, (cnn_mnist.input_dim_1_pool)*nb_filters_4])#-1*14*14*256

    else:
        flat_layer = tf.reshape(layer4_pool
                                , [-1, cnn_mnist.input_dim_4_pool*nb_filters_4])#-1*2*2*256

    return flat_layer, input_dim
