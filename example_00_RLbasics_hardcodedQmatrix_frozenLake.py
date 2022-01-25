#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 21:43:22 2021

@author: artur
"""

import pandas
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import tensorflow
import gym

font = {"family": "serif",
        "weight": "normal",
        "size": 18}

matplotlib.rc("font", **font)
matplotlib.rcParams["figure.figsize"] = (12, 9)
matplotlib.style.use("seaborn-colorblind")

# %% Train the agent.

# Create the environment.
env = gym.make('FrozenLake-v1')

# Initialize table with all zeros - rows are states, columns are actions.
Q = np.zeros([env.observation_space.n, env.action_space.n])

# Set learning parameters
lr = 0.8  # Relaxation for updating the Q values.
y = 0.95  # Discounting parameter for prioritising long- (1) or short-term (0) goals.
num_episodes = 2000

# Create lists to contain total rewards per episode.
rList = []

# Simulate every episode.
for i in range(num_episodes):
    # Reset environment and get first new observation
    s = env.reset()
    rAll = 0
    done = False
    j = 0
    # The Q-Table learning algorithm.
    while j < 99:
        j += 1
        # Choose an action by greedily (with noise) picking from Q table. Noise means that we cannot always perform the
        # optimum action.
        a = np.argmax(Q[s, :] + np.random.randn(1, env.action_space.n)*(1./(i+1)))
        # Get new state and reward from environment
        s1, r, done, _ = env.step(a)
        # Update Q-Table with new knowledge
        Q[s, a] = Q[s, a] + lr*(r + y*np.max(Q[s1, :]) - Q[s, a])
        rAll += r
        s = s1
        if done:
            break
    rList.append(rAll)

# %% Visualise the results using the trained agent.
obs = env.reset()
for i in range(1000):
    a = np.argmax(Q[s, :] + np.random.randn(1, env.action_space.n)*(1./(i+1)))
    obs, rewards, done, info = env.step(a)
    env.render()
    if done:
        break
