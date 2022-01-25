import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer - -1 will be replaced with batch size that comes from len(features)
    # (28x28) size of image, 1 channel for greyscale
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=32,
            kernel_size=[5, 5], # 5x5 window swept over the input layer
            padding="same", # pad sides with zeros to preserve the size of input layer
            activation=tf.nn.relu)
            # output size is (batch_size, 28, 28, 32) <- for each filter

    # Pooling Layer #1
    # group 2x2 pools without overlap with strides=2, output is (batch_size, 14, 14, 32)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    # output is (batch_size, 14, 14, 64)
    conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)

    # output is (batch_size, 7, 7, 64)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer - set inputs to be 2D (batch_size, 3316)
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    # add 1024 neurons
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    # drop 40% of neurons ONLY during training at random to avoid overfitting
    dropout = tf.layers.dropout(
            inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer - 10 outputs, one for each value, connect to dropouts
    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
            # Generate predictions (for PREDICT and EVAL mode)
            "classes": tf.argmax(input=logits, axis=1),
            # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
            # `logging_hook`.
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    # for prediction only, return the predictions
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    # activates logits with one-hot vectors for each element in a batch,
    # setting the index of the label to 1 and padding with zeros.
    # no loss would mean that prediciton returns 1.0 at the position of each label
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(
                    labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(unused_argv):
    modelDataDir = "./mnist_convnet_model"

    # Load training and eval data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    test_data = mnist.test.images # Returns np.array
    test_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    # visualise the training data
    # import matplotlib.pyplot as plt
    # for i in range(10):
    #     plt.figure()
    #     plt.title(train_labels[i])
    #     testPic = np.flipud(train_data[i,:].reshape((28,28), order="F").T)
    #     x, y = np.meshgrid(np.linspace(0, 1, 28), np.linspace(0, 1, 28))
    #     plt.contourf(x, y, testPic)
    # plt.show()

    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir=modelDataDir)

    # Set up logging for predictions to log probabilities computed during training
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data}, # data features
        y=train_labels, # ground truth
        batch_size=100, # 100 items to train on each step
        num_epochs=None, # train for a given number of iters
        shuffle=True) # random order

    mnist_classifier.train(
        input_fn=train_input_fn, # add input
        steps=20000, # fixed no. steps
        hooks=[logging_hook]) # connect logger

    # Evaluate accuracy of the model and print results
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": test_data},
        y=test_labels,
        num_epochs=1, # use all the test data
        shuffle=False) # keep fixed order
    test_results = mnist_classifier.evaluate(input_fn=test_input_fn)
    print(test_results)

if __name__ == "__main__":
    tf.app.run()
