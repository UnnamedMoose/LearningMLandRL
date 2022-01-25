import tensorflow as tf
import numpy as np
import collections
import os
# import argparse
import datetime as dt

def read_words(filename):
    """ Read the text data, replace end of lines with <eos> delimiter, following
    similar idea as with <unk> for uncommon words already there in the data.
    Assemble all the words into a single long list. """
    with tf.gfile.GFile(filename, "r") as f:
        return f.read().replace("\n", "<eos>").split()

def build_vocabulary(filename):
    """ Read the data and count frequency of each word. """
    # read the data into a single long list with all sentences in it
    data = read_words(filename)

    # count the occurrence of each word in the data, this includes delimiters
    counter = collections.Counter(data)

    # sort the words and their occurrences in descending order according to the
    # number of occurrences which is in the second element of each tuple; for
    # words with the same number of counts sort alphabetically
    # NOTE not sure if alphabetical order matters at all, if not can use simpler syntax
    # count_pairs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    # unpack the sorted words into a list
    words, _ = list(zip(*count_pairs))

    # give each word a unique integer identifier, with lowest IDs marking most
    # frequent words
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id

def file_to_word_ids(filename, word_to_id):
    """ Read the file and translate it from words to integers using word_to_id map """
    data = read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]

def load_data(data_path):
    # get the data paths
    train_path = os.path.join(data_path, "ptb.train.txt")
    valid_path = os.path.join(data_path, "ptb.valid.txt")
    test_path = os.path.join(data_path, "ptb.test.txt")

    # build the complete vocabulary, map words to integer IDs denoting
    # their frequency of occurence (low ID = frequent word)
    word_to_id = build_vocabulary(train_path)

    # read all the data sets and translate them from regular words to integer IDs
    train_data = file_to_word_ids(train_path, word_to_id)
    valid_data = file_to_word_ids(valid_path, word_to_id)
    test_data = file_to_word_ids(test_path, word_to_id)

    # number of words in the net's dictionary
    vocabulary = len(word_to_id)
    # map from IDs to words
    reversed_dictionary = dict(zip(word_to_id.values(), word_to_id.keys()))

    return train_data, valid_data, test_data, vocabulary, reversed_dictionary

def batch_producer(raw_data, batch_size, num_steps):
    """ Extracts x and y data from raw_data in the form of a tensor.
    num_steps is the number of unfolded time steps being fed into the network during
    training. """
    # convert data to a tensor
    raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

    data_len = tf.size(raw_data)
    batch_len = data_len // batch_size # no. full batches in the data
    # reshape the data as (batch_size, batch_len) for full batches only
    data = tf.reshape(raw_data[0: batch_size * batch_len],
                      [batch_size, batch_len])

    # set the epoch such that all of the data is passed over in one epoch;
    # this gives the number of time-step-sized batches that are available to be
    # iterated through in a single epoch.
    epoch_size = (batch_len - 1) // num_steps

    # set up a simple queue which allows the asynchronous and threaded extraction
    # of data batches from a pre-existing dataset.
    # each time more data is required in the training of the model, a new integer
    # is extracted between 0 and epoch_size.
    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()

    # extract x and y samples of data, with y being shifted by 1 index.
    # each row of the extracted x and y tensors will be an individual sample of
    # length num_steps and the number of rows is the batch length.
    # This keeps the sentence order preserved.
    x = data[:, i * num_steps:(i + 1) * num_steps]
    x.set_shape([batch_size, num_steps])
    y = data[:, i * num_steps + 1: (i + 1) * num_steps + 1]
    y.set_shape([batch_size, num_steps])
    return x, y

class Input(object):
    """ Defines input to the LST model """
    def __init__(self, batch_size, num_steps, data):
        self.batch_size = batch_size # size of each batch
        self.num_steps = num_steps # no. time steps
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        # convert the data to tensorsm with targets being the intended output
        self.input_data, self.targets = batch_producer(data, batch_size, num_steps)

# create the main model
class Model(object):
    """ Class definition for the RNN network model. """

    def __init__(self, input, is_training, hidden_size, vocab_size, num_layers,
                 dropout=0.5, init_scale=0.05):
        self.is_training = is_training
        self.input_obj = input
        self.batch_size = input.batch_size
        self.num_steps = input.num_steps
        self.hidden_size = hidden_size

        # create the word embeddings
        with tf.device("/cpu:0"):
            # create a lookup table for the words using as many columns as there
            # are neurons in the hidden layer; initialise with a uniform distribution
            # with values in range (-init_scale, init_scale).
            embedding_size = self.hidden_size
            embedding = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -init_scale, init_scale))
            # route the inputs to the LSTM coming from the input object through
            # the embedding lookup table to get the correct embedding vector that
            # will actually flow into the neural net.
            inputs = tf.nn.embedding_lookup(embedding, self.input_obj.input_data)

        # add a droput rate to avoid overfitting in the embedding layer
        if is_training and dropout < 1:
            inputs = tf.nn.dropout(inputs, dropout)

        # set up the state storage / extraction; this is used to reset state
        # variables to zero at the beginning of each epoch. During each epoch,
        # final state variables of the previous batch will be loaded as the initial
        # state.
        # size of the variable is 2 for h (output from previous LSTM cell) and
        # s (previous state variable); these two have the size equal to the hidden
        # layer, and we want to transfer values for each batch; furthermore, since
        # the net is stacked during training, we need num_layers for each stack.
        self.init_state = tf.placeholder(tf.float32, [num_layers, 2, self.batch_size, self.hidden_size])

        # unpack the initial state into num_layers x (2, batch_size, hidden_size)
        # tensors and transform them to a tupe that will be fed into TF LSTM cell
        state_per_layer_list = tf.unstack(self.init_state, axis=0)
        rnn_tuple_state = tuple(
            [tf.contrib.rnn.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1])
                for idx in range(num_layers)]
        )

        # create a basic LSTM cell to be unrolled.
        # forget bias values of 1.0 help guard against repeated low forget gate
        # outputs causing vanishing gradients.
        cell = tf.contrib.rnn.LSTMCell(hidden_size, forget_bias=1.0)

        # add a dropout wrapper if training
        if is_training and dropout < 1:
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout)

        # unroll the LSTM cell into the desired number of layers.
        if num_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell([cell for _ in range(num_layers)], state_is_tuple=True)

        # Create dynamic rnn object which will peform unrolling over each time step.
        # This is fed with the data coming out of the word embedding object.
        # Specify the initial state with the tf tuple.
        # State is the (s, h) tuple which will be fed to the next iteration.
        output, self.state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32, initial_state=rnn_tuple_state)

        # reshape to (batch_size * num_steps, hidden_size) to connect with
        # softmax output layer.
        output = tf.reshape(output, [-1, hidden_size])

        # create the softmax tensor multiplication variables (xw+b) and connect with the RNN output
        softmax_w = tf.Variable(tf.random_uniform([hidden_size, vocab_size], -init_scale, init_scale))
        softmax_b = tf.Variable(tf.random_uniform([vocab_size], -init_scale, init_scale))
        logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)

        # Reshape logits to be a 3-D tensor for sequence loss
        logits = tf.reshape(logits, [self.batch_size, self.num_steps, vocab_size])

        # set up a weighted cross entropy loss over a sequence of values.
        # targets tensor has a shape (batch_size, num_steps) with each value being
        # an integer corresponding to a unique word - contains the sequence LSTM is to predict.
        # The third argument is the weights tensor, of shape (batch_size, num_steps),
        # which allows weighting different samples or time steps with respect to the loss,
        # e.g. might want the loss to favor the latter time steps rather than the earlier ones.
        # No weighting is applied in this model, so a tensor of ones is passed to this argument.
        loss = tf.contrib.seq2seq.sequence_loss(
            logits, # connect inputs
            self.input_obj.targets, # sequence of target values
            tf.ones([self.batch_size, self.num_steps], dtype=tf.float32), # weights
            average_across_timesteps=False, # if yes, cost summed over the time step
            average_across_batch=True) # if yes, cost summed over the batch

        # Update the cost and reduce to a scalar variable
        self.cost = tf.reduce_sum(loss)

        # apply the softmax operator to the logits
        # TODO would make more sense to move logits defitnition here?
        self.softmax_out = tf.nn.softmax(tf.reshape(logits, [-1, vocab_size]))

        # set prediction to be the words (ints) with highest probability coming from the
        # softmax tensor output.
        self.predict = tf.cast(tf.argmax(self.softmax_out, axis=1), tf.int32)

        # compare predictions to the correct values and compute accuracy.
        correct_prediction = tf.equal(self.predict, tf.reshape(self.input_obj.targets, [-1]))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # that's it if not training
        # NOTE: need to get all the way here if evaluating, but for prediction only
        # could probably skip the loss?
        if not is_training:
           return

        # create a learning rate variable which will be adjusted over the duration of training.
        self.learning_rate = tf.Variable(0.0, trainable=False)

        # get trainable variables.
        tvars = tf.trainable_variables()

        # apply a cap to gradients to a maximum of 5 units to avoid diminishing gradients
        # during backpropagation.
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), 5)

        # create a descent optimiser which uses the learning rate.
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        # optimizer = tf.train.AdamOptimizer(self.learning_rate)

        # feed the capped gradients to the optimiser.
        self.train_op = optimizer.apply_gradients(zip(grads, tvars),
            global_step=tf.contrib.framework.get_or_create_global_step())
        # self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)

        # create a placeholder which will accept a new learning rate from the feed dict.
        self.new_lr = tf.placeholder(tf.float32, shape=[])
        # push the new learning rate to the model (will be called at the start of
        # each new epoch)
        self.lr_update = tf.assign(self.learning_rate, self.new_lr)

    def assign_lr(self, session, lr_value):
        """ Separate method for updating the learning rate on an instance of model class. """
        session.run(self.lr_update, feed_dict={self.new_lr: lr_value})

def train(train_data, vocabulary, num_layers, num_epochs, batch_size, model_save_name,
    model_save_dir, learning_rate=1.0, max_lr_epoch=10, lr_decay=0.93, print_iter=50):
    """ Method which trains the RNN. """

    # setup data for the model; num_steps is the size of unfolded network
    training_input = Input(batch_size=batch_size, num_steps=35, data=train_data)

    # create an instance of the RNN model.
    m = Model(training_input, is_training=True, hidden_size=650, vocab_size=vocabulary,
          num_layers=num_layers)

    # initialisation op created once the tf graph structure has been established in
    # the Model::__init__ method.
    init_op = tf.global_variables_initializer()

    # hold initial decay of learning rate
    orig_decay = lr_decay

    # start a new session for training
    with tf.Session() as sess:
        # initialise
        sess.run([init_op])

        # create a thread coordinator to manage queuing inside the Input class.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # initialise saving of progress
        saver = tf.train.Saver()

        # train the net for a given number of epochs
        for epoch in range(num_epochs):
            # update decay of learning rate and update the value
            new_lr_decay = orig_decay ** max(epoch + 1 - max_lr_epoch, 0.0)
            m.assign_lr(sess, learning_rate * new_lr_decay)

            # create initial state populated with zeros - this is updated as the
            # optimiser moves through each epoch.
            current_state = np.zeros((num_layers, 2, batch_size, m.hidden_size))

            # get the time
            curr_time = dt.datetime.now()

            # feed the time-step sized training batches to the net
            for step in range(training_input.epoch_size):
                # cost, _ = sess.run([m.cost, m.optimizer])
                # For each training batch, run the cost, training, and state operations.
                # Take the output of the m.state operation and then feed it to the net
                # on the next pass using the feed_dict as init_state.
                if step % print_iter != 0:
                    cost, _, current_state = sess.run([m.cost, m.train_op, m.state],
                                                      feed_dict={m.init_state: current_state})
                else:
                    seconds = (float((dt.datetime.now() - curr_time).seconds) / print_iter)
                    curr_time = dt.datetime.now()
                    cost, _, current_state, acc = sess.run([m.cost, m.train_op, m.state, m.accuracy],
                                                       feed_dict={m.init_state: current_state})
                    print("Epoch {}, Step {}, cost: {:.3f}, accuracy: {:.3f}, Seconds per step: {:.3f}".format(
                        epoch, step, cost, acc, seconds))
                    print("Current learning rate {:6.4e}, decay {:6.4e}\n".format(m.learning_rate.eval(), new_lr_decay))

            # save a model checkpoint after eac epoch
            saver.save(sess, os.path.join(model_save_dir, model_save_name), global_step=epoch)

        # do a final save
        saver.save(sess, os.path.join(model_save_dir, "{}_final".format(model_save_name)))

        # close threads
        coord.request_stop()
        coord.join(threads)

def test(model_path, test_data, reversed_dictionary):
    """ Evaluation method which imports a saved model and checks its accuracy. """
    # establish an input
    test_input = Input(batch_size=20, num_steps=35, data=test_data)

    # create a model
    m = Model(test_input, is_training=False, hidden_size=650, vocab_size=vocabulary,
              num_layers=2)

    # create a saver
    saver = tf.train.Saver()

    # start a new session
    with tf.Session() as sess:
        # start threads
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # initialise the state of the net
        current_state = np.zeros((2, 2, m.batch_size, m.hidden_size))

        # restore the trained model
        saver.restore(sess, model_path)

        # get an average accuracy over num_acc_batches
        num_acc_batches = 30
        check_batch_idx = 25 # check output at this iteration
        # warm-up the model to establish state vars before starting to measure accuracy
        acc_check_thresh = 5
        accuracy = 0

        for batch in range(num_acc_batches):
            if batch == check_batch_idx:
                # run evaluation
                true_vals, pred, current_state, acc = sess.run([m.input_obj.targets, m.predict, m.state, m.accuracy],
                                                               feed_dict={m.init_state: current_state})
                pred_string = [reversed_dictionary[x] for x in pred[:m.num_steps]]
                true_vals_string = [reversed_dictionary[x] for x in true_vals[0]]
                print("True values (1st line) vs predicted values (2nd line):")
                print(" ".join(true_vals_string))
                print(" ".join(pred_string), "\n")

            else:
                acc, current_state = sess.run([m.accuracy, m.state], feed_dict={m.init_state: current_state})

            if batch >= acc_check_thresh:
                accuracy += acc

        print("Average accuracy: {:.3f}".format(accuracy / (num_acc_batches-acc_check_thresh)))

        # close threads
        coord.request_stop()
        coord.join(threads)

# ===
if __name__ == "__main__":
    data_path = "../data/ptbTextData"
    model_save_dir="./lstmModel"

    # read the data as lists of integer IDs; reversed_dictionary maps IDs to words
    train_data, valid_data, test_data, vocabulary, reversed_dictionary = load_data(data_path)

    # train the model
    train(train_data, vocabulary, num_layers=2, num_epochs=60, batch_size=20,
          model_save_name="two-layer-lstm-medium-config-60-epoch-0p93-lr-decay-10-max-lr",
          model_save_dir=model_save_dir)

    # test the model
    # # read the last checkpoint number
    # trained_model = tf.train.latest_checkpoint(model_save_dir, latest_filename=None)
    # # restore the model and run the evaluation function
    # test(trained_model, test_data, reversed_dictionary)
