"""A simple TensorFlow Ops"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

a = tf.constant(2.5, name='first_val')
b = tf.constant(2.5, name='Second_val')

sum = a+b

'''print(tf.get_default_graph().get_operations())
print(tf.get_default_graph().get_tensor_by_name('first_val:0'))

tf.train.write_graph(tf.get_default_graph(
), 'D:/Amr.Soliman/Documents/GitHub/tensorflow-for-dummies/', 'graph.dat')'''

with tf.Session() as sess:
    res1 = sess.run(a)
    print(res1)
