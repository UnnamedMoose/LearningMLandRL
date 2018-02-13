# ====
# NOTE: this is how you can visualise what's saved in the model
# from tensorflow.python.tools import inspect_checkpoint as chkp
# latestModelVersion = tf.train.latest_checkpoint(modelPath, latest_filename=None)
# print("All")
# chkp.print_tensors_in_checkpoint_file(latestModelVersion, tensor_name="",
#     all_tensors=True, all_tensor_names=True)

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# import the function that defines the estimator model
from mnistNet_0_train import cnn_model_fn

modelPath = "./mnist_convnet_model"

# create the estimator, will automatically restore parameters from the latest checkpoint
mnist_classifier = tf.estimator.Estimator(model_dir=modelPath, model_fn=cnn_model_fn)

# Load test data
mnistData = tf.contrib.learn.datasets.load_dataset("mnist")
test_data = mnistData.test.images
test_labels = np.asarray(mnistData.test.labels, dtype=np.int32)

# define the input function
inputFunction = tf.estimator.inputs.numpy_input_fn(
    x={"x":test_data},
    y=None,
    num_epochs=1,
    shuffle=False)

# run the predictions
predictions = mnist_classifier.predict(input_fn=inputFunction)

# show a couple of images
iShown = -1
for pred, expect in zip(predictions, test_labels):
    iShown += 1
    if iShown > 10:
        break

    # prepare the test picture
    testPic = np.flipud(test_data[iShown,:].reshape((28,28), order="F").T)
    x, y = np.meshgrid(np.linspace(0, 1, 28), np.linspace(0, 1, 28))

    # plot
    fig, axarr = plt.subplots(1, 2, figsize=(6,4))
    plt.suptitle("This should be a {:d} and we think it is a {:d}".format(expect, pred["classes"]))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None)

    axarr[0].xaxis.set_major_locator(plt.NullLocator())
    axarr[0].yaxis.set_major_locator(plt.NullLocator())
    axarr[0].pcolor(x, y, testPic, cmap="Greys")
    axarr[0].set_aspect('equal')

    possibleGueses = np.sort(np.unique(test_labels))
    colours = ["k" for i in range(len(pred["probabilities"]))]
    colours[pred["classes"]] = "r"
    axarr[1].bar(possibleGueses, pred["probabilities"], 0.4, color=colours)
    axarr[1].set_xlabel("Number")
    axarr[1].set_ylabel("Confidence")
    axarr[1].xaxis.set_ticks(possibleGueses)

plt.show()
