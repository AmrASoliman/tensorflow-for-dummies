from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import os
x = tf.constant([1])
y = tf.constant([1])
m = tf.Variable(tf.random_normal([]))
b = tf.Variable(tf.random_normal([]))
model = tf.add(tf.multiply(x, m), b)
loss = tf.reduce_mean(tf.pow(model-y, 2))
