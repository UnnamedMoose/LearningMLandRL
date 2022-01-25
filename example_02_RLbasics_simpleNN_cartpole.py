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
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

font = {"family": "serif",
        "weight": "normal",
        "size": 18}

matplotlib.rc("font", **font)
matplotlib.rcParams["figure.figsize"] = (12, 9)
matplotlib.style.use("seaborn-colorblind")

# Consider using: tf_upgrade_v2 --infile example_02_RLbasics_simpleNN_cartpole.py --outfile example_02_RLbasics_simpleNN_cartpole_v2.py

# %% Create the environment.
env = gym.make('CartPole-v0')

gamma = 0.99
modelPath = "../modelData"
modelName = "cartpole_v0"

tf.compat.v1.disable_eager_execution()


def save_frames_as_mp4(frames, path='../Figures/animations', filename='gym_animation.mp4'):

    # Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    Writer = animation.writers["ffmpeg"]
    writer = Writer(fps=60, metadata=dict(artist="Artur"), bitrate=1800)
    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save(os.path.join(path, filename), writer=writer)


def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


class agent():
    def __init__(self, lr, s_size, a_size, h_size):
        # These lines established the feed-forward part of the network. The agent takes a state and produces an action.
        self.state_in = tf.compat.v1.placeholder(shape=[None, s_size], dtype=tf.float32)
        # hidden = slim.fully_connected(self.state_in, h_size, biases_initializer=None, activation_fn=tf.nn.relu)
        # self.output = slim.fully_connected(hidden, a_size, activation_fn=tf.nn.softmax, biases_initializer=None)
        hidden = tf.compat.v1.layers.dense(self.state_in, h_size, bias_initializer=None, activation=tf.nn.relu)
        self.output = tf.compat.v1.layers.dense(hidden, a_size, activation=tf.nn.softmax, bias_initializer=None)
        self.chosen_action = tf.argmax(self.output, 1)

        # The next six lines establish the training proceedure. We feed the reward and chosen action into the network
        # to compute the loss, and use it to update the network.
        self.reward_holder = tf.compat.v1.placeholder(shape=[None], dtype=tf.float32)
        self.action_holder = tf.compat.v1.placeholder(shape=[None], dtype=tf.int32)

        self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder
        self.responsible_outputs = tf.compat.v1.gather(tf.reshape(self.output, [-1]), self.indexes)

        self.loss = -tf.reduce_mean(tf.compat.v1.log(self.responsible_outputs)*self.reward_holder)

        tvars = tf.compat.v1.trainable_variables()
        self.gradient_holders = []
        for idx, var in enumerate(tvars):
            placeholder = tf.compat.v1.placeholder(tf.float32, name=str(idx)+'_holder')
            self.gradient_holders.append(placeholder)

        self.gradients = tf.gradients(self.loss, tvars)

        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders, tvars))


# Clear the Tensorflow graph.
tf.compat.v1.reset_default_graph()

# Load the agent.
myAgent = agent(lr=1e-2, s_size=4, a_size=2, h_size=8)

total_episodes = 2500
max_ep = 1999
update_frequency = 5

init = tf.compat.v1.global_variables_initializer()

# Launch the tensorflow graph
with tf.compat.v1.Session() as sess:
    sess.run(init)

    # Add ops to save and then restore all the variables.
    saver = tf.compat.v1.train.Saver()

    # Initialise episode counters and storage for outputs.
    i = 0
    total_reward = []
    total_length = []

    gradBuffer = sess.run(tf.compat.v1.trainable_variables())
    for ix, grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0

    while i < total_episodes:
        s = env.reset()
        running_reward = 0
        ep_history = []
        for j in range(max_ep):
            # Probabilistically pick an action given our network outputs.
            a_dist = sess.run(myAgent.output, feed_dict={myAgent.state_in: [s]})
            a = np.random.choice(a_dist[0], p=a_dist[0])
            a = np.argmax(a_dist == a)

            # Get our reward for taking an action given a bandit.
            s1, r, d, _ = env.step(a)
            ep_history.append([s, a, r, s1])
            s = s1
            running_reward += r
            if d:
                # Update the network.
                ep_history = np.array(ep_history)
                ep_history[:, 2] = discount_rewards(ep_history[:, 2])
                feed_dict = {myAgent.reward_holder: ep_history[:, 2],
                             myAgent.action_holder: ep_history[:, 1],
                             myAgent.state_in: np.vstack(ep_history[:, 0])}
                grads = sess.run(myAgent.gradients, feed_dict=feed_dict)
                for idx, grad in enumerate(grads):
                    gradBuffer[idx] += grad

                if i % update_frequency == 0 and i != 0:
                    feed_dict = dictionary = dict(zip(myAgent.gradient_holders, gradBuffer))
                    _ = sess.run(myAgent.update_batch, feed_dict=feed_dict)
                    for ix, grad in enumerate(gradBuffer):
                        gradBuffer[ix] = grad * 0

                total_reward.append(running_reward)
                total_length.append(j)
                break

            # Update our running tally of scores.
        if i % 100 == 0:
            print("Average score after episode {:d} = {:.2f}".format(i, np.mean(total_reward[-100:])))
        i += 1

    # Save the modified variables to disk.
    modelPath = os.path.join(modelPath, modelName)
    try:
        os.mkdir(modelPath)
    except FileExistsError:
        pass
    save_path = saver.save(sess, os.path.join(modelPath, modelName))

# %% Visualise the results.
fig, ax = plt.subplots()
fig.canvas.set_window_title("Total reward")
plt.plot(total_reward)

fig, ax = plt.subplots()
fig.canvas.set_window_title("Total balancing length [no. time steps]")
plt.plot(total_length)

# %% Visualise the results using the trained agent.

# Create a saver for loading the model.
saver = tf.compat.v1.train.Saver()

frames = []

with tf.compat.v1.Session() as sess:
    saver.restore(sess, os.path.join(modelPath, modelName))

    obs = env.reset()
    for i in range(2000):
        a_dist = sess.run(myAgent.output, feed_dict={myAgent.state_in: [obs]})
        a = np.random.choice(a_dist[0], p=a_dist[0])
        a = np.argmax(a_dist == a)
        obs, rewards, done, info = env.step(a)
        frames.append(env.render(mode="rgb_array"))
        if done:
            print("done after", i, "iterations")
            # break

env.close()
save_frames_as_mp4(frames, filename="{}.mp4".format(modelName))
