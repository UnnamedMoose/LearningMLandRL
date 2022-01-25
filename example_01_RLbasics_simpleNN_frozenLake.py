#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 21:43:22 2021

@author: artur
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import gym
import tensorflow as tf

font = {"family": "serif",
        "weight": "normal",
        "size": 18}

matplotlib.rc("font", **font)
matplotlib.rcParams["figure.figsize"] = (12, 9)
matplotlib.style.use("seaborn-colorblind")

# %% Create the environment.
env = gym.make('FrozenLake-v1')

# %% Creat the NN.
modelPath = "../modelData"
modelName = "model_frozenLake_basic"

# Run in V1 to avoid rewriting all of the code for now.
tf.compat.v1.disable_v2_behavior()

tf.compat.v1.reset_default_graph()

# Establish the feed-forward part of the network used to choose actions.
inputs1 = tf.compat.v1.placeholder(shape=[1, 16], dtype=tf.float32)
W = tf.Variable(tf.random.uniform([16, 4], 0, 0.01))
Qout = tf.matmul(inputs1, W)
predict = tf.argmax(Qout, 1)

# Set up computation of the loss fuction by taking the sum of squares difference between the target and predicted Q values.
nextQ = tf.compat.v1.placeholder(shape=[1, 4], dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1)
updateModel = trainer.minimize(loss)

"""
# NOTE: almost compatible with v2...
# Establish the feed-forward part of the network used to choose actions.
inputs1 = tf.keras.Input(shape=[1, 16], dtype=tf.float32, name="feedFwdInput")
W = tf.Variable(tf.random.uniform([16, 4], 0, 0.01))
Qout = tf.matmul(inputs1, W)
predict = tf.argmax(Qout, 1)

# Set up computation of the loss fuction by taking the sum of squares difference between the target and predicted Q values.
nextQ = tf.keras.Input(shape=[1, 4], dtype=tf.float32, name="nextQinp")
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.optimizers.SGD(learning_rate=0.1)
updateModel = trainer.minimize(loss, var_list=[W], tape=tf.GradientTape())\
"""

# %% Train the network.
init = tf.compat.v1.initialize_all_variables()

# Set learning parameters
y = 0.99
e = 0.1
num_episodes = 2000

# Create lists to contain total rewards and steps per episode
jList = []
rList = []
with tf.compat.v1.Session() as sess:
    # Initialise the session.
    sess.run(init)

    # Add ops to save and then restore all the variables.
    saver = tf.compat.v1.train.Saver()

    # Run all the episodes for training.
    for i in range(num_episodes):
        print("Evaluating episode {:d}".format(i))

        # Reset environment and get first new observation
        s = env.reset()
        rAll = 0
        done = False
        j = 0
        # Take a fixed no. time steps.
        while j < 99:
            j += 1
            # Choose an action by greedily (with e chance of random action) from the Q-network
            a, allQ = sess.run([predict, Qout], feed_dict={inputs1: np.identity(16)[s:s+1]})
            if np.random.rand(1) < e:
                a[0] = env.action_space.sample()
            # Get new state and reward from environment
            s1, r, done, _ = env.step(a[0])
            # Obtain the Q' values by feeding the new state through our network
            Q1 = sess.run(Qout, feed_dict={inputs1: np.identity(16)[s1:s1+1]})
            # Obtain maxQ' and set our target value for chosen action.
            maxQ1 = np.max(Q1)
            targetQ = allQ
            targetQ[0, a[0]] = r + y*maxQ1
            # Train the network using target and predicted Q values
            _, W1 = sess.run([updateModel, W], feed_dict={inputs1: np.identity(16)[s:s+1], nextQ: targetQ})
            rAll += r
            s = s1
            if done:
                # Reduce chance of random action as we train the model.
                e = 1./((i/50) + 10)
                break
        jList.append(j)
        rList.append(rAll)

    # Save the modified variables to disk.
    modelPath = os.path.join(modelPath, modelName)
    try:
        os.mkdir(modelPath)
    except FileExistsError:
        pass
    save_path = saver.save(sess, os.path.join(modelPath, modelName))

print("Percent of succesful episodes: {:.2f}%".format(sum(rList)/num_episodes*100))

# %% Visualise the results.
fig, ax = plt.subplots()
fig.canvas.set_window_title("Succesful episodes")
plt.plot(rList)

fig, ax = plt.subplots()
fig.canvas.set_window_title("No. steps per episode")
plt.plot(jList)

# %% Visualise the results using the trained agent.

# Create a saver for loading the model.
saver = tf.compat.v1.train.Saver()

with tf.compat.v1.Session() as sess:
    # Restore variables from disk.
    saver.restore(sess, os.path.join(modelPath, modelName))

    obs = env.reset()
    for i in range(1000):
        a, allQ = sess.run([predict, Qout], feed_dict={inputs1: np.identity(16)[obs:obs+1]})
        obs, rewards, done, info = env.step(a[0])
        env.render()
        if done:
            break
