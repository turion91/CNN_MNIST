# -*- coding: utf-8 -*-
"""Contains the variables to be used.
"""

__author__ = 'Adriano Vereno'

__status__ = 'Development'
__version__ = '0.0.1'
__date__ = '2018-02-06'



#path to files
#path is a boolean that specifies True for fashion and False for digit MNIST
path = True
if path == True:    
    data_path = '/home/turion91/Desktop/fashion_police/'
else:   
    data_path = '/home/turion91/Desktop/fashion_police/digit/'
#path for tensoboard    
tensorboard_path = "/home/turion91/Desktop/fashion_police/mnist/"
#path for the output folder
folder = data_path + 'output/'
#path for the result csv
result_path = data_path + 'output/out_frame.csv'
#path for the corrected result csv
reult_path_corrected = data_path + 'output/out_frame_corrected.csv'


#weight stdandard deviation initialisation
w_std = 0.1
#bias constant initialisation
b_cont = 0.1
#filter size
filter_size = 5
#conv filter stride
conv_strides = [1, 1, 1, 1]

#pooling layer size
ksize = [1, 2, 2, 1]
#pooling layer stride
pool_strides = [1, 2, 2, 1]

nb_channels = 1#one channel color

#nb of filters used per conv layers
nb_filters_1 = 32
nb_filters_2 = 64
nb_filters_3 = 128
nb_filters_4 = 256

#nb of neurons per hidden layers
rate = 0.5
nb_hidden_1 = [1024, 700]
nb_hidden_2 = [int(nb_hidden_1[0]*rate)]
nb_hidden_3 = [int(nb_hidden_2[0]*rate)]
nb_hidden_4 = [int(nb_hidden_3[0]*rate)]

#nb of labels
nb_labels = 10

#1st label to keep
keep_label = 0

#2nd label to keep
keep_label_2 = 9

#original shape
original_shape = 28*28
#shape after 4 pooling layers
input_dim_4_pool = 2*2
#shape after 3 pooling layers
input_dim_3_pool = 4*4
#shape after 2 pooling layers
input_dim_2_pool = 7*7
#shape after 1 pooling layers
input_dim_1_pool = 14*14
#shape with no pooling layers
input_dim_0_pool = 28*28
#nb of batches
nb_batches = 50
#batch size
batch_size = 50

#boolean for presence or absence of pooling layer
pool_conv1 = True
pool_conv2 = True
pool_conv3 = True
pool_conv4 = True

#boolean for presence or absence of dropout
drop_conv1 = False
drop_conv2 = False
drop_conv3 = False
drop_conv4 = False

drop_dense1 = True
drop_dense2 = True
drop_dense3 = True
drop_dense4 = True

#nb of conv layers
nb_conv = [2]

#dense layer setting
nb_dense = [1]

#learning rate
learning_rate = 1e-2

#max pooling or avg pooling: True = Max and False = avg
max_pooling = True
