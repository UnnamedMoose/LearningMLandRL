import os
import tensorflow as tf

modelPath = "./modelData"
modelName = "model.ckpt"

# Create some variables.
v1 = tf.get_variable("var1", shape=[3])
v2 = tf.get_variable("var2", shape=[5])

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
    # Restore variables from disk.
    saver.restore(sess, os.path.join(modelPath, modelName))

    # Check the values of the variables
    print("Model restored with variables")
    print("var1 : %s" % v1.eval())
    print("var2 : %s" % v2.eval())
