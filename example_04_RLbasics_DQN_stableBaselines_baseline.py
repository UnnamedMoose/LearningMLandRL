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
import pyglet
import datetime
import shutil

import stable_baselines3
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
# from stable_baselines3.common import set_global_seeds
# from stable_baselines.common.evaluation import evaluate_policy
# from stable_baselines.common.cmd_util import make_vec_env
from stable_baselines3.common import results_plotter

font = {"family": "serif",
        "weight": "normal",
        "size": 18}

matplotlib.rc("font", **font)
matplotlib.rcParams["figure.figsize"] = (12, 9)
matplotlib.style.use("seaborn-colorblind")

# %% Functions.


def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        return env
    # set_global_seeds(seed)
    return _init


def evaluate_agent(model, env, num_episodes=100, num_steps=None, num_last_for_reward=None, render=False):
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
        if num_steps is None:
            num_steps = 1000000
        for i in range(num_steps):
            # _states are only useful when using LSTM policies
            action, _states = model.predict(obs)
            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            obs, reward, done, info = env.step(action)
            episode_rewards.append(reward)
            if render:
                frames.append(env.render(mode="rgb_array"))
            if done:
                break

        if num_last_for_reward is None:
            all_episode_rewards.append(sum(episode_rewards))
        else:
            all_episode_rewards.append(np.mean(episode_rewards[-num_last_for_reward:], 1))

    mean_episode_reward = np.mean(all_episode_rewards)
    print("Mean reward:", mean_episode_reward, "Num episodes:", num_episodes)

    if render:
        return frames, mean_episode_reward
    else:
        return mean_episode_reward


def save_frames_as_mp4(frames, path='./Figures', filename='gym_animation.mp4'):

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


# %% Main.

if __name__ == "__main__":
    # envName = 'CartPole-v0'
    envName = 'LunarLander-v2'

    nProc = 8  # Number of processes to use

    # modelName = "DQN"
    modelName = "A2C"

    # Set up constants etc.
    saveFile = "modelData/{}_{}".format(envName, modelName)
    logDir = "modelData/{}_{}_logs".format(envName, modelName)
    os.makedirs(logDir, exist_ok=True)
    models = {
        "A2C": stable_baselines3.A2C,
        "DQN": stable_baselines3.DQN
    }

    # Use separate environements for training and evaluation
    # env = gym.make(envName)
    env = SubprocVecEnv([make_env(envName, i) for i in range(nProc)])
    eval_env = gym.make(envName)
    env = VecMonitor(env, logDir)

    # Create the model using stable baselines.
    # model = DQN("MlpPolicy", env, learning_rate=1e-3, verbose=0)
    model = models[modelName]("MlpPolicy", env, learning_rate=1e-3, verbose=0)

    # Random Agent, before training
    print("Before training")
    mean_reward = evaluate_agent(model, eval_env, num_episodes=1, num_steps=10000)

    # Train the agent for N steps
    starttime = datetime.datetime.now()
    model.learn(total_timesteps=200000, log_interval=10)
    endtime = datetime.datetime.now()
    print("Training took {:.0f} seconds".format((endtime-starttime).total_seconds()))

    # Store the log for comparisons.
    shutil.rmtree(logDir+"_firstTrainingBatch")
    shutil.copytree(logDir, logDir+"_firstTrainingBatch")

    # Save the agent.
    model.save(saveFile)

    # Load and evaluate the trained agent
    trained_model = models[modelName].load(saveFile)
    print("After training")
    mean_reward = evaluate_agent(trained_model, eval_env, num_episodes=1, num_steps=10000)

    # Reconnect the agent with the environment.
    trained_model.set_env(env)

    # Train the agent for N more steps
    starttime = datetime.datetime.now()
    trained_model.learn(total_timesteps=200000, log_interval=10)
    endtime = datetime.datetime.now()
    print("Training took {:.0f} seconds".format((endtime-starttime).total_seconds()))

    # Evaluate again
    print("After some more training")
    mean_reward = evaluate_agent(trained_model, eval_env, num_episodes=1, num_steps=10000)

# %% Test the trained model and visualise the results.

    # Reset the evironment for a new simulation and run it to make a movie.
    try:
        state = env.reset()
        frames, mean_reward = evaluate_agent(model, eval_env, render=True, num_episodes=1)
        print("Finished final test episode, reward={:.2f}".format(mean_reward))

        env.close()

        # Save renders as an animation.
        save_frames_as_mp4(frames, filename="{}_{}_stableBaselines.mp4".format(envName, modelName))

    except pyglet.canvas.xlib.NoSuchDisplayException:
        print("Rendering failed, probably running on a node with no graphics?")

# %% Plot results.
    results_plotter.plot_results([logDir], num_timesteps=1e15, x_axis=results_plotter.X_EPISODES,
                                 task_name="",figsize=(10, 8))
    fig, ax = plt.gcf(), plt.gca()
    fig.canvas.set_window_title("Test restarts")
    plt.tight_layout()

    df_0 = results_plotter.load_results(logDir+"_firstTrainingBatch")
    xy_list = results_plotter.ts2xy(df_0, results_plotter.X_EPISODES)

    x, y = xy_list[0], xy_list[1]
    plt.plot(x, y, "gs", ms=5, mew=1, mfc="None", alpha=0.5)
    # Do not plot the smoothed curve at all if the timeseries is shorter than window size.
    if x.shape[0] >= results_plotter.EPISODES_WINDOW:
        # Compute and plot rolling mean with window of size EPISODE_WINDOW
        x, y_mean = results_plotter.window_func(x, y, results_plotter.EPISODES_WINDOW, np.mean)
        plt.plot(x, y_mean, "g-", alpha=0.2, lw=4)

