import pandas as pd
import numpy as np
import tensorflow as tf
import random
import sys
from data import data

epochs = 1000
sample_size = 400
x_train, y_train, x_test, y_test = data(sample_size)
feature_number = x_train.shape[1]
hidden_number = 5

with tf.name_scope("input"):
    x = tf.placeholder('float',[None,feature_number])
    x_ = tf.placeholder('float',[None,feature_number])
    y = tf.placeholder('float',[None,1])
    y_ = tf.placeholder('float',[None,1])

    w0 = tf.Variable(tf.random_normal([1], 0.1, 0.1))
    w = tf.Variable(tf.random_normal([feature_number, 1], 0.05, 0.01))
    v = tf.Variable(tf.random_normal([hidden_number,feature_number], 0.05, 0.05))

    y_hat = tf.Variable(tf.zeros([sample_size,1]))

with tf.name_scope("fm"):
    linear_terms = tf.add(w0,tf.reduce_sum(tf.matmul(x, w),1,keep_dims=True)) # n * 1
    pair_interactions = 0.5 * tf.reduce_sum(
        tf.subtract(
            tf.pow(
                tf.matmul(x,tf.transpose(v)),2),
            tf.matmul(tf.pow(x,2),tf.transpose(tf.pow(v,2)))
        ),axis = 1 , keep_dims=True)

    y_hat = tf.add(linear_terms,pair_interactions)

with tf.name_scope("train"):
    #y_hat__ = tf.sigmoid(y_hat)
    #y__ = tf.sigmoid(y)
    loss = tf.losses.mean_squared_error(y, y_hat)
    #loss = tf.losses.log_loss(y, y_hat)
    train_op = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(loss)

with tf.name_scope("eval"):
    linear_terms_test = tf.add(w0,tf.reduce_sum(tf.matmul(x_, w),1,keep_dims=True)) # n * 1
    pair_interactions_test = 0.5 * tf.reduce_sum(
        tf.subtract(
            tf.pow(
                tf.matmul(x_,tf.transpose(v)),2),
            tf.matmul(tf.pow(x_,2),tf.transpose(tf.pow(v,2)))
        ),axis = 1 , keep_dims=True)

    y_hat_test = tf.add(linear_terms_test,pair_interactions_test)
    predictions_test = tf.to_int32(y_hat_test)
    y_ = tf.to_int32(y_)
    corrections_test = tf.equal(predictions_test, y_)
    accuracy_test = tf.reduce_mean(tf.cast(corrections_test, tf.float32))

    predictions_train = tf.to_int32(y_hat)
    y_int = tf.to_int32(y)
    corrections_train = tf.equal(predictions_train, y_int)
    accuracy_train = tf.reduce_mean(tf.cast(corrections_train, tf.float32))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for times in range(epochs):
        _, l, a_test, a_train = sess.run([train_op, loss, accuracy_test, accuracy_train], feed_dict={x: x_train, y: y_train, x_: x_test, y_: y_test})
        print("epochs:", times, "loss:", l, "accuracy_test:", a_test, "accuracy_train:", a_train)