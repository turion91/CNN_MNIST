# -*- coding: utf-8 -*-
"""Contains the main running script.
"""

__author__ = 'Adriano Vereno'

__status__ = 'Development'
__version__ = '0.0.1'
__date__ = '2018-02-06'

import os
import fileinput
import csv
import time
import numpy as np
import pandas as pd
from datetime import timedelta
import variables
import CNN_layers as cnn
import Dense_layers as dense
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from itertools import product 


from numpy.random import seed
seed(42)
tf.set_random_seed(42)
#Import nist data from your folder. WARNING you need to have the fashion MNIST
#data in the same format as the MNIST digit, else tensorflow will download
#the MNIST digit and work on it which is not what you want
#you can find all the data in the right format here: https://github.com/zalandoresearch/fashion-mnist

#path is the variable that makes the switch between fashion and mnist
path = variables.path

#change the boolean to fashion or digit for easier comprehension
if path == True:
    path = 'Fashion'
else:
    path = 'Digit' 

#try to access the folder of the MNIST digits, if it does not exist, it will be created    
try:
    data = input_data.read_data_sets(variables.data_path, one_hot=True)
except:
    os.makedirs((variables.data_path))
    data = input_data.read_data_sets(variables.data_path, one_hot=True)   


#path to tensorboard    
tensor_path = variables.tensorboard_path
#output path for results
folder_out = variables.folder
#path for the output csv
output_path = variables.result_path
#path for the corrected output csv (removes empty rows)
output_corrected_path = variables.reult_path_corrected

filter_size = variables.filter_size
nb_channels = variables.nb_channels
nb_filters_1 = variables.nb_filters_1
nb_filters_2 = variables.nb_filters_2
nb_filters_3 = variables.nb_filters_3
nb_filters_4 = variables.nb_filters_4

nb_hidden_1 = variables.nb_hidden_1
nb_hidden_2 = variables.nb_hidden_2
nb_hidden_3 = variables.nb_hidden_3
nb_hidden_4 = variables.nb_hidden_4
nb_labels = variables.nb_labels

keep_label_1 = variables.keep_label
keep_label_2 = variables.keep_label_2

original_shape = variables.original_shape
input_dim_4_pool = variables.input_dim_4_pool
input_dim_3_pool = variables.input_dim_3_pool
input_dim_2_pool = variables.input_dim_2_pool
input_dim_1_pool = variables.input_dim_1_pool
input_dim_0_pool = variables.input_dim_0_pool

nb_batches = variables.nb_batches
batch_size = variables.batch_size

pool_conv1 = variables.pool_conv1
pool_conv2 = variables.pool_conv2
pool_conv3 = variables.pool_conv3
pool_conv4 = variables.pool_conv4

drop_conv1 = variables.drop_conv1
drop_conv2 = variables.drop_conv2
drop_conv3 = variables.drop_conv3
drop_conv4 = variables.drop_conv4

drop_dense1 = variables.drop_dense1
drop_dense2 = variables.drop_dense2
drop_dense3 = variables.drop_dense3
drop_dense4 = variables.drop_dense4

nb_conv = variables.nb_conv
nb_dense = variables.nb_dense
learning_rate = variables.learning_rate

#dict to store variables
res = []

for d, c, neurons_1, neurons_2, neurons_3, neurons_4 in product(nb_dense, nb_conv
                                                             , nb_hidden_1, nb_hidden_2, nb_hidden_3, nb_hidden_4):
    tf.reset_default_graph()#reseet the graph each time
    with tf.Graph().as_default():#set the tensorflow graph
        tf.set_random_seed(42)#seed for reproducibility
        #create the placeholders for x and y
        #calling the variables
        out = {}
    
        x = tf.placeholder(tf.float32, shape=[None, original_shape])#? 28*28
        y_ = tf.placeholder(tf.float32, shape=[None, nb_labels])#? 10
        keep_prob = tf.placeholder(tf.float32)#placeholder for dropout
        x_image = tf.reshape(x, [-1, 28, 28, 1])#image reshaping
        if c == 1:
            conv_layer, input_dimension = cnn.cp1(x_image, keep_prob)#calling the conv function
        elif c == 2:
            conv_layer, input_dimension = cnn.cp2(x_image, keep_prob)#calling the conv function
        elif c == 3:
            conv_layer, input_dimension = cnn.cp3(x_image, keep_prob)#calling the conv function
        else:
            conv_layer, input_dimension = cnn.cp4(x_image, keep_prob)#calling the conv function
    
        if d == 1:
            y_conv = dense.dense1(c, conv_layer, neurons_1
                                  , input_dimension, keep_prob)#calling the dense function
        elif d == 2:
            y_conv = dense.dense2(c, conv_layer, neurons_1, neurons_2
                                  , input_dimension, keep_prob)#calling the dense function
        elif d == 3:
            y_conv = dense.dense3(c, conv_layer, neurons_1, neurons_2, neurons_3
                                  , input_dimension, keep_prob)#calling the dense function
        else:
            y_conv = dense.dense4(c, conv_layer, neurons_1, neurons_2, neurons_3, neurons_4
                                  , input_dimension, keep_prob)#calling the dense function
        #creates the loss function needed to be optimized
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    
#        for l in learning_rate:
        #will use the adamOptimizer with a slow learning rate to optimize the loss function
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("cost", cross_entropy)
        tf.summary.scalar("accuracy", accuracy)
        summary_op = tf.summary.merge_all()
        with tf.Session() as sess:
            if d == 1:
                writer = tf.summary.FileWriter(tensor_path + 'nb_conv_'+ str(c)+'nb_hidden_'+ str(d)+'_nb_neurons_1_'+str(neurons_1)
            , graph=tf.get_default_graph())
            elif d == 2:
                writer = tf.summary.FileWriter(tensor_path + 'nb_conv_'+ str(c)+'nb_hidden_'+ str(d)+'_nb_neurons_1_'+str(neurons_1)
            +'_nb_neurons_2_'+str(neurons_2), graph=tf.get_default_graph())
            elif d == 3:
                writer = tf.summary.FileWriter(tensor_path + 'nb_conv_'+ str(c)+'nb_hidden_'+ str(d)+'_nb_neurons_1_'+str(neurons_1)
            +'_nb_neurons_2_'+str(neurons_2)+'_nb_neurons_3_'+str(neurons_3), graph=tf.get_default_graph())
            else:
                writer = tf.summary.FileWriter(tensor_path + 'nb_conv_'+ str(c)+'nb_hidden_'+ str(d)+'_nb_neurons_1_'+str(neurons_1)
            +'_nb_neurons_2_'+str(neurons_2)+'_nb_neurons_3_'+str(neurons_3)+'_nb_neurons_4_'+str(neurons_4), graph=tf.get_default_graph())
                
            sess.run(tf.global_variables_initializer())#Initialize all the previous variables
            start_time = time.time()
            for i in range(nb_batches):
                batch = data.train.next_batch(batch_size)#Work in batch of 50 to not overload the memory
                
                batch_xs = []
                batch_ys = []
                for x1, x2 in zip(batch[0], batch[1]):
                    numbers = np.argmax(x2)
                    if numbers == keep_label_1:
                        batch_ys.append(x2)
                        batch_xs.append(x1)
                    if numbers == keep_label_2:
                        batch_xs.append(x1)
                        batch_ys.append(x2)
                batch_xs = np.vstack(batch_xs)
                batch_ys = np.vstack(batch_ys)
                batch_ys = batch_ys[:,[keep_label_1, 5]]
                
                
                if nb_labels == 2:
                    if i % 5 == 0 or i == nb_batches-1:
                        train_accuracy = accuracy.eval(feed_dict={
                            x: batch_xs, y_: batch_ys, keep_prob: 1.0})
                        test_acc = accuracy.eval(feed_dict={x: data.test.images, y_: data.test.labels, keep_prob: 1.0})
                        out['test_acc_'+str(i)], out['train_acc_'+str(i)] = test_acc, train_accuracy
                        print('step %d, training accuracy %g' % (i, train_accuracy))#Compute and print train accuracy
                    _, summary = sess.run([train_step, summary_op], feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})#dropout of 0.5
                    writer.add_summary(summary, i)
                    
                else:
                    if i % 5 == 0 or i == nb_batches-1:
                        train_accuracy = accuracy.eval(feed_dict={
                            x: batch[0], y_: batch[1], keep_prob: 1.0})
                        test_acc = accuracy.eval(feed_dict={x: data.test.images, y_: data.test.labels, keep_prob: 1.0})
                        out['test_acc_'+str(i)], out['train_acc_'+str(i)] = test_acc, train_accuracy
                        print('step %d, training accuracy %g' % (i, train_accuracy))#Compute and print train accuracy
                    _, summary = sess.run([train_step, summary_op], feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})#dropout of 0.5
                    writer.add_summary(summary, i)
                    
            if nb_labels == 2:
                test_xs = []
                test_ys = []
                for x1, x2 in zip(data.test.images, data.test.labels):
                    numbers = np.argmax(x2)
                    if numbers == keep_label_1:
                        test_ys.append(x2)
                        test_xs.append(x1)
                    if numbers == keep_label_2:
                        test_xs.append(x1)
                        test_ys.append(x2)
                test_xs = np.vstack(test_xs)
                test_ys = np.vstack(test_ys)
                test_ys = test_ys[:,[keep_label_1, 5]]
                accu = accuracy.eval(feed_dict={
                    x: test_xs, y_: test_ys, keep_prob: 1.0})
                print('test accuracy %g' % accu)#Compute the test accuracy no dropout
            
            else:
                accu = accuracy.eval(feed_dict={
                    x: data.test.images, y_: data.test.labels, keep_prob: 1.0})
                print('test accuracy %g' % accu)#Compute the test accuracy no dropout
            # Ending time.
    
            end_time = time.time()
    
            # Difference between start and end-times.
    
            time_dif = end_time - start_time
    
            # Print the time-usage.
    
            print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
            if d == 1:
                print('nb_conv: '+ str(c), 'nb_dense: ' + str(d), 'nb_neurons_1_layer: ' + str(neurons_1))
            elif d == 2:
                print('nb_conv: '+ str(c), 'nb_dense: ' + str(d), 'nb_neurons_1_layer: ' + str(neurons_1)
                , 'nb_neurons_2_layer: ' + str(neurons_2))
            elif d == 3:
                print('nb_conv: '+ str(c), 'nb_dense: ' + str(d), 'nb_neurons_1_layer: ' + str(neurons_1)
                , 'nb_neurons_2_layer: ' + str(neurons_2), 'nb_neurons_3_layer: ' + str(neurons_3))
            else:
                print('nb_conv: '+ str(c), 'nb_dense: ' + str(d), 'nb_neurons_1_layer: ' + str(neurons_1)
                , 'nb_neurons_2_layer: ' + str(neurons_2), 'nb_neurons_3_layer: ' + str(neurons_3), 'nb_neurons_4_layer: ' + str(neurons_4)) 

            #store all the wanted variables in the out dict
            out['nb_conv'] = c
            out['nb_dense'] = d
            out['nb_neurons_1'] = neurons_1
            if d == 2 or d == 3 or d == 4:
                out['nb_neurons_2'] = neurons_2
            else:
                out['nb_neurons_2'] = 'This network does not have enough layers'
            if d == 3 or d == 4:
                out['nb_neurons_3'] = neurons_3
            else:
                out['nb_neurons_3'] = 'This network does not have enough layers'
            if d == 4:
                out['nb_neurons_4'] = neurons_4
            else:
                out['nb_neurons_4'] = 'This network does not have enough layers'
                
            out['Time taken'] = time_dif
            out['Test accuracy'] = accu
            out['Learning rate'] = learning_rate
            out['Batch size'] = batch_size
            out['nb_batches'] = nb_batches
            out['nb_classes'] = nb_labels
            out['1st label kept'] = keep_label_1
            out['2nd label kept'] = keep_label_2
            out['Filter size'] = filter_size
            out['Fashion or Digit'] = path
            #append the dict to res and make a dataframe out of it
            res.append(out)

            
keys = res[0].keys()
out_frame = pd.DataFrame(res)
#try to convert the dataframe to csv and append each run to the same one
#if the path does not exist, it will be created. Skip duplicate rows
#and removes empty ones
try:
    out_frame.to_csv(output_path, sep=",", mode='a', index=False)
    seen = set()
    for line in fileinput.FileInput(output_path, inplace=1):
        if line in seen: continue
        seen.add(line)
        print (line, )
    with open(output_path) as input, open(output_corrected_path, 'w', newline='') as output:
        writer = csv.writer(output)
        for row in csv.reader(input):
            if any(field.strip() for field in row):
                writer.writerow(row)
except:
    #if an error occurs about the directory existing already, pass
    try:
        os.makedirs((folder_out))
    except OSError as e:
        if e.errno != os.errno.EEXIST:
            raise   
        pass
    out_frame.to_csv(output_path, sep=",", mode='a', index=False)
    seen = set()
    for line in fileinput.FileInput(output_path, inplace=1):
        if line in seen: continue
        seen.add(line)
        print (line, )
    with open(output_path) as input, open(output_corrected_path, 'w', newline='') as output:
        writer = csv.writer(output)
        for row in csv.reader(input):
            if any(field.strip() for field in row):
                writer.writerow(row)
                    
                
