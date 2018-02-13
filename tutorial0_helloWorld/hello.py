#!/usr/bin/python3

# default TensorFlow intallation comes without support for advanced CPU instructions;
# disable the related warnings as we're not doing-high end stuff so don't care (yet).
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

# ===
# define a constant
hello = tf.constant('Hello, TensorFlow!\n', name="my_first_constant")

with tf.Session() as sess:
    # execute the constant in the session which evaluates it; print the result
    print(sess.run(hello))

# ===
# define some more constants, none of these are actually initialised yet
x = tf.constant([[37.0, -23.0], [1.0, 4.0]])
w = tf.Variable(tf.random_uniform([2, 2]))
y = tf.matmul(x, w)
output = tf.nn.softmax(y)
init_op = w.initializer

with tf.Session() as sess:
    # Run the initializer on `w`.
    sess.run(init_op)

    # Evaluate `output`. `sess.run(output)` will return a NumPy array containing
    # the result of the computation.
    print(sess.run(output))

    # Evaluate `y` and `output`. Note that `y` will only be computed once, and its
    # result used both to return `y_val` and as an input to the `tf.nn.softmax()`
    # op. Both `y_val` and `output_val` will be NumPy arrays.
    y_val, output_val = sess.run([y, output])
    print(y_val, output_val, "\n")

# ===
