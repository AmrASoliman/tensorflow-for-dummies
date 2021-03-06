"""A simple TensorFlow Ops"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
"""
tf.logging.set_verbosity(tf.logging.INFO)
t1 = tf.constant(5.5)

with tf.Session() as sess:
    output = sess.run(t1)
    tf.logging.info('Output: %f', output) """

a = tf.constant(2.5)
b = tf.constant(4.5)
total = a+b

tf.summary.scalar("a", a)
tf.summary.scalar("b", b)
tf.summary.scalar("total", total)

merged_op = tf.summary.merge_all()

writer = tf.summary.FileWriter("summary")

with tf.Session() as sess:
    summary = sess.run(merged_op)
    writer.add_summary(summary)
    writer.close()
