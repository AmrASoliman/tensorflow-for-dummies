"""A simple TensorFlow Ops"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

t1 = tf.constant([[1., 1.], [1., 2.], [1., 1.]])
t2 = tf.constant([[1., 1.], [1., 2.]])

#dot_0 = tf.tensordot(t1, t2, 0)
#dot_1 = tf.tensordot(t1, t2, 1)
t_mat = tf.matmul(t1, t2)

with tf.Session() as sess:
   # print(sess.run(dot_0))
    print('_______________________')
   # print(sess.run(dot_1))
    print('_________________________')
    print(sess.run(t_mat))
