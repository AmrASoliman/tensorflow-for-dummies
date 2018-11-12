"""A simple TensorFlow Ops"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf


t1 = tf.constant([1., 2., 3.])
t2 = tf.constant([4., 5., 6.])

t3 = tf.squared_difference(t1, t2)

t4 = tf.rsqrt(t2)

t5 = tf.pow(t2, t1)
t6 = tf.exp(t2)

with tf.Session() as sess:
    print(sess.run(t6))
