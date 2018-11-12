"""A simple TensorFlow application"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

# Create tensor
""" msg = tf.string_join(["Hello ", "TensorFlow"])
t1 = tf.ones([3, 3])
t0 = tf.zeros([2, 2])
t2 = tf.fill([3, 2, 5], 5.2) """

lin_tensor = tf.lin_space(0., 10., 6)
range_tensor = tf.range(7., delta=0.5)

ren_ints = tf.random_normal([1000000], dtype=tf.float64, mean=20)

vec = tf.constant([1, 2, 3, 4, 5, 6])
mat = tf.reshape(vec, [3, 2])

# Lunch Session
with tf.Session() as sess:
    print(sess.run(vec))
    print('__________________')
    print(sess.run(mat))
    ''' print(sess.run(msg))
     print(sess.run(t0))
     print(sess.run(t1))
    print(sess.run(t2))'''
