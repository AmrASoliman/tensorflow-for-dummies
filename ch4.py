from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import os

# enable Looging
tf.logging.set_verbosity(tf.logging.INFO)

# Create Tensors
t1 = tf.constant([1.2, 2.3, 3.4, 4.5])
t2 = tf.constant([5.6, 6.7, 7.8, 8.9])
t3 = tf.concat([t1, t2], 0)
t4 = tf.random_normal([8])
t5 = tf.tensordot(t3, t4, 1)

# Create Summery data
tf.summary.scalar("t1", t1)
tf.summary.scalar("t2", t2)
tf.summary.scalar("t3", t3)
tf.summary.scalar("t4", t4)
tf.summary.scalar("t5", t5)
merged_op = tf.summary.merge_all()

# create FileWriter
file_writer = tf.summary.FileWriter("log", graph=tf.get_default_graph())

# execute first graph
with tf.Session() as sess:
    # excute the session
    dot_result, summary = sess.run(t5, merged_op)

    # write the result to the log
    tf.logging.info('Result of Dot product: %f', dot_result)

    # print thew Summery Data
    file_writer.add_summary(summary)
    file_writer.flush()

    # obtain the graphDef and write to a file
    tf.train.write_graph(sess.graph, os.getcwd(), 'graph1.dat')

# create Second Graph and make it default
graph = tf.Graph()
with graph.as_default():

    # compute the average
    t6 = tf.random_uniform([8], 4.0, 8.0)
    t7 = tf.fill([8], 6.0)
    t8 = tf.reduce_mean(t6, t7)

    # execute first graph
    with tf.Session() as sess:
        # ecxute the session
        sess.run(t8)

        # optain the graphdef and write it to file
        tf.train.write_graph(sess, os.getcwd(), 'graph2.dat')
