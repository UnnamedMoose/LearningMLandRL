import os
import tensorflow as tf

# where to save the model to
modelPath = "./modelData"
# name of the model files; NOTE: this is just the prefix of the files, they will
# be automatically appended with global steps etc. as the model evolves
modelName = "model.ckpt"

# Create some variables.
v1 = tf.get_variable("v1", shape=[3], initializer = tf.zeros_initializer)
v2 = tf.get_variable("v2", shape=[5], initializer = tf.zeros_initializer)

inc_v1 = v1.assign(v1+1)
dec_v2 = v2.assign(v2-1)

# Add an op to initialize the variables.
init_op = tf.global_variables_initializer()

# Add ops to save and restore the chosen variables (empty arg list for all vars)
# can also change names of the variables in the checkpoint
saver = tf.train.Saver({"var1": v1, "var2": v2})

# Later, launch the model, initialize the variables, do some work, and save the
# variables to disk.
with tf.Session() as sess:
    sess.run(init_op)

    # Do some work with the model.
    inc_v1.op.run()
    dec_v2.op.run()

    # print the state of the variables being saved
    print("Saving model with values")
    print("v1 : %s" % v1.eval())
    print("v2 : %s" % v2.eval())

    # Save the modified variables to disk.
    save_path = saver.save(sess, os.path.join(modelPath, modelName))
    print("Model saved in path: %s" % save_path)
