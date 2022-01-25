#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 21:43:22 2021

@author: artur
"""

import tensorflow as tf
import tensorflow.keras as keras
import collections
import numpy as np
import random
import gym
import pickle
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

font = {"family": "serif",
        "weight": "normal",
        "size": 18}

matplotlib.rc("font", **font)
matplotlib.rcParams["figure.figsize"] = (12, 9)
matplotlib.style.use("seaborn-colorblind")

# %% Configuration paramaters for the whole setup

# How many hidden layers.
n_hidden_layers = 3

# How many neurons in each hidden layer.
layer_size = 256

# Reward discount rate.
gamma = 0.9

# After how many training cycles to update the target network.
update_freq = 10

# The CartPole-v0 is considered solved if for N consecutive episodes, the cart pole has not
# fallen over and it has achieved an average reward of 195.0.
# A reward of +1 is provided for every timestep the pole remains upright.
win_episodes = 100
win_reward = 195.0

# Max no. episodes to use for training.
episode_count = 3000

# How many episodes to use for exploration.
episodes_epsilon = 500

# How many frames to use in the experience replay buffer.
batch_size = 64

# Unique ID of this config.
modelName = "cartpole_dqn_v0"

# Set output path.
modelPath = "../modelData"
outdir = os.path.join(modelPath, modelName)

# Change here to train the agent again.
train = False

# %% Create the environment.

# in CartPole-v0:
#    action=0 is left and action=1 is right
#    state = [pos, vel, theta, angular speed]
env = gym.make('CartPole-v0')

# By default, CartPole-v0 has max episode steps = 200
env._max_episode_steps = 250

# Get the no. state variables.
state_size = env.observation_space.shape[0]

# %% Functions and classes.


class DQNAgent:
    def __init__(self, state_space, action_space, layer_size=256, n_layers=3, gamma=0.9, update_freq=10,
                 episodes_epsilon=500, outdir="."):
        """DQN Agent on CartPole-v0 environment

        Arguments:
            state_space (tensor): state space
            action_space (tensor): action space
        """
        self.layer_size = layer_size
        self.n_layers = n_layers

        # After how many training cycles to update the target network.
        self.update_frequency = update_freq

        self.action_space = action_space

        # experience buffer
        self.memory = []

        # discount rate
        self.gamma = gamma

        # initially 90% exploration, 10% exploitation
        self.epsilon = 1.0
        # iteratively applying decay til
        # 10% exploration/90% exploitation
        self.epsilon_min = 0.1
        self.epsilon_decay = self.epsilon_min / self.epsilon
        self.epsilon_decay = self.epsilon_decay**(1. / float(episodes_epsilon))

        # Q Network weights filename
        self.weights_file = os.path.join(outdir, "dqn_cartpole.h5")

        # Q Network for training
        n_inputs = state_space.shape[0]
        n_outputs = action_space.n
        self.q_model = self.build_model(n_inputs, n_outputs)
        self.q_model.compile(loss='mse', optimizer=keras.optimizers.Adam())
        # target Q Network
        self.target_q_model = self.build_model(n_inputs, n_outputs)
        # copy Q Network params to target Q Network
        self.update_weights()

        self.replay_counter = 0

    def save_weights(self):
        """save Q Network params to a file"""
        self.q_model.save_weights(self.weights_file)

    def load_weights(self):
        # TODO not tested
        """load Q Network params to a file"""
        self.q_model.load_weights(self.weights_file)

    def update_weights(self):
        """copy trained Q Network params to target Q Network"""
        self.target_q_model.set_weights(self.q_model.get_weights())

    def update_epsilon(self):
        """decrease the exploration, increase exploitation"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def remember(self, state, action, reward, next_state, done):
        """store experiences in the replay buffer

        Arguments:
            state (tensor): env state
            action (tensor): agent action
            reward (float): reward received after executing
                action on state
            next_state (tensor): next state
        """
        item = (state, action, reward, next_state, done)
        self.memory.append(item)

    def build_model(self, n_inputs, n_outputs):
        """Q Network is N-N-N MLP

        Arguments:
            n_inputs (int): input dim
            n_outputs (int): output dim

        Return:
            q_model (Model): DQN
        """
        inputs = keras.layers.Input(shape=(n_inputs, ), name='state')
        # x = Dense(self.layer_size, activation='relu')(inputs)
        x = inputs
        for i in range(self.n_layers):
            x = keras.layers.Dense(self.layer_size, activation='relu')(x)
        # x = Dense(self.layer_size, activation='relu')(x)
        # x = Dense(self.layer_size, activation='relu')(x)
        x = keras.layers.Dense(n_outputs, activation='linear', name='action')(x)
        q_model = keras.models.Model(inputs, x)
        q_model.summary()
        return q_model

    def act(self, state):
        """epsilon-greedy policy

        Return:
            action (tensor): action to execute
        """
        if np.random.rand() < self.epsilon:
            # explore - do random action
            return self.action_space.sample()

        # exploit
        q_values = self.q_model.predict(state)
        # select the action with max Q-value
        action = np.argmax(q_values[0])
        return action

    def get_target_q_value(self, next_state, reward):
        """compute Q_max

        Arguments:
            reward (float): reward received after executing action on state
            next_state (tensor): next state

        Return:
            q_value (float): max Q-value computed
        """
        # max Q value among next state's actions
        # DQN chooses the max Q value among next actions
        # selection and evaluation of action is
        # on the target Q Network
        # Q_max = max_a' Q_target(s', a')
        q_value = np.amax(self.target_q_model.predict(next_state)[0])
        q_value = q_value*self.gamma + reward
        return q_value

    def replay(self, batch_size):
        """ Replay experiences from the collected buffer. This addresses the correlation issue
            between consecutive samples.

        Arguments:
            batch_size (int): replay buffer batch sample size
        """
        # sars = state, action, reward, state' (next_state)
        sars_batch = random.sample(self.memory, batch_size)
        state_batch, q_values_batch = [], []

        # fixme: for speedup, this could be done on the tensor level
        # but easier to understand using a loop
        for state, action, reward, next_state, done in sars_batch:
            # policy prediction for a given state
            q_values = self.q_model.predict(state)

            # get Q_max
            q_value = self.get_target_q_value(next_state, reward)

            # correction on the Q value for the action used
            q_values[0][action] = reward if done else q_value

            # collect batch state-q_value mapping
            state_batch.append(state[0])
            q_values_batch.append(q_values[0])

        # train the Q-network
        self.q_model.fit(np.array(state_batch),
                         np.array(q_values_batch),
                         batch_size=batch_size,
                         epochs=1,
                         verbose=0)

        # update exploration-exploitation probability
        self.update_epsilon()

        # copy new params on old target after every N training updates
        if self.replay_counter % self.update_frequency == 0:
            self.update_weights()

        self.replay_counter += 1


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


# %% Train the model.
if train:
    # stores the reward per episode
    scores = collections.deque(maxlen=win_episodes)

    # Create a monitor for retrieving extra info.
    gym.logger.setLevel(gym.logger.ERROR)
    env = gym.wrappers.Monitor(env, directory=outdir, video_callable=False, force=True)
    env.seed(0)

    # instantiate the agent.
    agent = DQNAgent(env.observation_space, env.action_space, layer_size=layer_size, n_layers=n_hidden_layers, gamma=gamma,
                     update_freq=update_freq, episodes_epsilon=episodes_epsilon, outdir=outdir)

    # Q-Learning sampling and fitting
    for episode in range(episode_count):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        done = False
        total_reward = 0
        while not done:
            # in CartPole-v0, action=0 is left and action=1 is right
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            # in CartPole-v0:
            # state = [pos, vel, theta, angular speed]
            next_state = np.reshape(next_state, [1, state_size])
            # store every experience unit in replay buffer
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        # call experience relay
        if len(agent.memory) >= batch_size:
            agent.replay(batch_size)

        print("Finished episode {:d}, reward={:.2f}".format(episode, total_reward))

        scores.append(total_reward)
        mean_score = np.mean(scores)
        if mean_score >= win_reward and episode >= win_episodes:
            print("Solved in {:d} episodes, mean score = {:.2f}. Epsilon = {:.2f}".format(episode, mean_score, agent.epsilon))
            agent.save_weights()
            break
        if (episode + 1) % win_episodes == 0:
            print("Episode {:d}, mean score = {:.2f}".format((episode + 1), mean_score))

    # Close the env and write monitor result info to disk
    env.close()

    # Save the agent for future use.
    with open(os.path.join(outdir, "trained_agent.pkl"), "wb") as outfile:
        pickle.dump(agent, outfile)

# %% Test the trained model and visualise the results.

# Load the agent.
with open(os.path.join(outdir, "trained_agent.pkl"), "rb") as infile:
    trained_agent = pickle.load(infile)

# Reset the evironment for a new simulation.
state = env.reset()

# Reshape the state to match the network needs.
# TODO is this necessary?
state = np.reshape(state, [1, state_size])

# Perform a fixed no. timesteps.
done = False
total_reward = 0
frames = []
states = []
actions = []

for timestep in range(200):
    # Get the new action and reward.
    action = trained_agent.act(state)
    next_state, reward, done, _ = env.step(action)
    next_state = np.reshape(next_state, [1, state_size])
    total_reward += reward

    # Store data for visualisation.
    states.append(state)
    actions.append(action)
    frames.append(env.render(mode="rgb_array"))

    # Move to the next state.
    state = next_state

print("Finished final test episode, reward={:.2f}".format(total_reward))

env.close()

# Save renders as an animation.
save_frames_as_mp4(frames, filename="{}.mp4".format(modelName))
