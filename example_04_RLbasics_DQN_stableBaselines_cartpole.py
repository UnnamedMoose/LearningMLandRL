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
from stable_baselines3 import DQN
import pyglet
import datetime

font = {"family": "serif",
        "weight": "normal",
        "size": 18}

matplotlib.rc("font", **font)
matplotlib.rcParams["figure.figsize"] = (12, 9)
matplotlib.style.use("seaborn-colorblind")

# %% Create the environment.

# in CartPole-v0:
#    action=0 is left and action=1 is right
#    state = [pos, vel, theta, angular speed]
env = gym.make('CartPole-v0')

# Create the model using stable baselines.
model = DQN("MlpPolicy", env, verbose=0)

def evaluate_agent(model, env, num_episodes=100, render=False):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_episodes: (int) number of episodes to evaluate it
    :return: (float) Mean reward for the last num_episodes
    """
    frames = []

    # This function will only work for a single Environment
    all_episode_rewards = []
    for i in range(num_episodes):
        episode_rewards = []
        done = False
        obs = env.reset()
        while not done:
            # _states are only useful when using LSTM policies
            action, _states = model.predict(obs)
            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            obs, reward, done, info = env.step(action)
            episode_rewards.append(reward)
            if render:
                frames.append(env.render(mode="rgb_array"))

        all_episode_rewards.append(sum(episode_rewards))

    mean_episode_reward = np.mean(all_episode_rewards)
    print("Mean reward:", mean_episode_reward, "Num episodes:", num_episodes)

    if render:
        return frames, mean_episode_reward
    else:
        return mean_episode_reward


# Use a separate environement for evaluation
eval_env = gym.make('CartPole-v1')

# Random Agent, before training
print("Before training")
mean_reward = evaluate_agent(model, eval_env, num_episodes=100)

# Train the agent for N steps
starttime = datetime.datetime.now()
model.learn(total_timesteps=100000, log_interval=10)
endtime = datetime.datetime.now()
print("Training took {:.0f} seconds".format((endtime-starttime).total_seconds()))

# Evaluate the trained agent
print("After training")
mean_reward = evaluate_agent(model, eval_env, num_episodes=100)

# %% Test the trained model and visualise the results.


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


# Reset the evironment for a new simulation and run it to make a movie.
try:
    state = env.reset()
    frames, mean_reward = evaluate_agent(model, eval_env, render=True, num_episodes=1)
    print("Finished final test episode, reward={:.2f}".format(mean_reward))
    
    env.close()
    
    # Save renders as an animation.
    save_frames_as_mp4(frames, filename="{}.mp4".format("DQN_stableBaselines"))

except pyglet.canvas.xlib.NoSuchDisplayException:
    print("Rendering failed, probably running on a node with no graphics?")
