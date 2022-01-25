import numpy as np
import tensorflow as tf


# ===
# define some constant tensors
a = tf.constant(3.0, dtype=tf.float32)
b = tf.constant(4.0) # also tf.float32 implicitly
total = a + b

# create placeholders for future values
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
z = x + y

# print summary of the tensors (haven't been run yet so not intialised)
print(a)
print(b)
print(total)

with tf.Session() as sess:
    # run one tensor and print to screen
    print(sess.run(total), "\n")

    # execute a number of tensors in the same call
    print(sess.run({'ab':(a, b), 'total':total}), "\n")

    # execute an addition operation with different parameters fed to the placeholders
    print(sess.run(z, feed_dict={x: [1, 3], y: [2, 4.5]}), "\n")

    # can use this to override values of any pre-made tensors
    print(sess.run(total, feed_dict={a: 1, b: 2}), "\n")

# ===
# define some arrayed data
arrData = [
    [0, 1,],
    [2, 3,],
    [4, 5,],
    [6, 7,],
]
# create a dataset instance
slices = tf.data.Dataset.from_tensor_slices(arrData)
# create an iterator for the data which will retrieve the next row on each call
next_item = slices.make_one_shot_iterator().get_next()

with tf.Session() as sess:
    # retrieve all rows from the arrayed data
    while True:
        try:
            print(sess.run(next_item))
        except tf.errors.OutOfRangeError:
            print("No more data.\n")
            break

# ===
# create a placeholder for a batch of 3-element vectors
x = tf.placeholder(tf.float32, shape=[None, 3])
# create a dense layer
linear_model = tf.layers.Dense(units=1)
# assign the batch of vectors to the layer
y = linear_model(x)
# create an initialiser that will populate the weights of the layer
# NOTE: I think the weights will be random
init = tf.global_variables_initializer()

with tf.Session() as sess:
    # run the initialiser
    sess.run(init)

    # run the linear model
    print(sess.run(y, {x: [[1, 2, 3],[4, 5, 6]]}), "\n")

# ===
# define a simple database of features
features = {
    'sales' : [[5], [10], [8], [9]],
    'department': ['sports', 'sports', 'gardening', 'gardening']}

# create a categorical column of the different departments
department_column_cat = tf.feature_column.categorical_column_with_vocabulary_list(
        'department', ['sports', 'gardening'])
# turn the categorical column into a dense column to feed to input layer;
# this creates a one-hot vector for each department
department_column = tf.feature_column.indicator_column(department_column_cat)

# create two columns, with the sales being a simple numeric column
columns = [
    tf.feature_column.numeric_column('sales'),
    department_column
]

# generate an inputlayer of the features and their matching data columns
inputs = tf.feature_column.input_layer(features, columns)

# create initialisers for the variables and lookup tables (needed by categorical columns)
var_init = tf.global_variables_initializer()
table_init = tf.tables_initializer()

# create a writer for graph visualisation; produces an "event" file
writer = tf.summary.FileWriter("./")
writer.add_graph(tf.get_default_graph())

with tf.Session() as sess:
    # initialise
    sess.run((var_init, table_init))

    # run the inputs layer
    print(sess.run(inputs), "\n")
