import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# correlation variable x and ground truth values
x = tf.constant([[1], [2], [3], [4]], dtype=tf.float32, name="inputData")
y_true = tf.constant([[0], [-1], [-2], [-3]], dtype=tf.float32, name="groundTruth")

# linear model
linear_model = tf.layers.Dense(units=1, name="regressionModel")

# prediciton of y from x using the linear model is what we're after
y_pred = linear_model(x)

# loss model that computes mean square error between the ground truth and predictions
loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)

# create an optimiser instance that will tune the coefficients of the graph
# in order to minimise the loss function
optimizer = tf.train.GradientDescentOptimizer(0.01, name="gradientOpt")
train = optimizer.minimize(loss)

# initialiser
init = tf.global_variables_initializer()

# create a writer for graph visualisation; produces an "event" file
writer = tf.summary.FileWriter("./modelData")
writer.add_graph(tf.get_default_graph())

lossTrace = []
with tf.Session() as sess:
    # initialise
    sess.run(init)

    # run the training
    for i in range(1000):
        _, loss_value = sess.run((train, loss))
        lossTrace = np.append(lossTrace, loss_value)
        if i%100 == 0:
            print("Iter {:2d}, loss ={:7.4f}".format(i, loss_value))

    # come up with the final prediciton and convert to numpy arrays for further processing
    independentVariable = sess.run(x)
    finalPred = sess.run(y_pred)
    groundTruth = sess.run(y_true)
    print("\nFinal prediction: {}".format(finalPred))
    print("\nGround truth: {}".format(groundTruth))

plt.figure()
plt.plot(lossTrace, "kp--", ms=5, lw=2)
plt.xlabel("Iteration")
plt.ylabel("Loss [-]")

plt.figure()
plt.plot(independentVariable, groundTruth, "kp--", ms=9, lw=2, label="Ground truth")
plt.plot(independentVariable, finalPred, "rx--", ms=9, lw=2,
    markeredgewidth=2, label="Predicted value")
plt.legend(prop={"size":14})
plt.xlabel("x")
plt.ylabel("y")

plt.show()
