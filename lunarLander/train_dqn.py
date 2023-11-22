# Author: Simon Guldager Andersen
# Date (latest update): 

### SETUP ------------------------------------------------------------------------------------

## Imports:
import os
import sys
import time
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt


import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rcParams
from cycler import cycler

import gymnasium as gym

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback


## Change directory to current one
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

## Set plotting style and print options
sns.set_theme()
sns.set_style("darkgrid")
sns.set_context("paper") #Possible are paper, notebook, talk and poster

d = {'lines.linewidth': 2, 'axes.titlesize': 18, 'axes.labelsize': 18, 'xtick.labelsize': 12, 'ytick.labelsize': 12,\
     'legend.fontsize': 15, 'font.family': 'serif', 'figure.figsize': (9,6)}
d_colors = {'axes.prop_cycle': cycler(color = ['teal', 'navy', 'coral', 'plum', 'purple', 'olivedrab',\
         'black', 'red', 'cyan', 'yellow', 'khaki','lightblue'])}
rcParams.update(d)
rcParams.update(d_colors)
np.set_printoptions(precision = 5, suppress=1e-10)


### FUNCTIONS ----------------------------------------------------------------------------------

def test_many_models(model_list, Neval, env):


    # time it
    start_time = time.time()
    for model_path in model_list:
        model = DQN.load(model_path, env=env)
        inter_time = time.time()

        print("load time: ,", inter_time-start_time)
       # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=Neval)

        # preidct
        obs = env.reset()
        rewards_list = []
        Ngames = 0
        k = 0
        steps_list = []
        while Ngames < Neval:
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            k += 1
            if dones:
                rewards_list.append(rewards)
                obs = env.reset()
                Ngames += 1
                print(f"Game {Ngames} finished after {k} steps")
                steps_list.append(k)
                k = 0
        print("mean steps and std: ", np.mean(steps_list), np.std(steps_list))
        mean_reward = np.mean(rewards_list)
        std_reward = np.std(rewards_list)

        end_time = time.time()
        print("Time per episode: ", (end_time-inter_time)/Neval)
        print(f"Model {model_path} mean_reward:{mean_reward:.2f} +/- {std_reward}")
    return

def predict_many_models(model_list, model_numbers, Neval, env):
    # for each model ,predict actions Neval times

    # test model
    Ncomplete = 0

    Nmodels = len(model_list)
    results = np.zeros([Nmodels, 3])
    results[:,-1] = model_numbers

    for i, model_path in enumerate(model_list):
        model = DQN.load(model_path, env=env)

        
        obs = env.reset()
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=Neval)
        results[i,0] = mean_reward
        results[i,1] = std_reward
        print(f"Model {model_path} mean_reward:{mean_reward:.2f} +/- {std_reward}")
    return results

def find_best_results(base_model_path, min_mean, max_std, ddof = 1):
    npz_path = os.path.join(base_model_path, 'log', 'evaluations.npz')

    npz = np.load(npz_path)
    results = npz['results']
    steps = npz['timesteps']

    means, stds = results.mean(axis = 1), results.std(axis = 1, ddof = ddof)

    mask_mean = means > min_mean
    mask_std = stds < max_std

    mask = ((mask_mean) & (mask_std))
    npz.close()

    return steps[mask]


### MAIN ---------------------------------------------------------------------------------------

def main():
    
    train_model, load_model, test_many_models = False, False, True

    pretrained_model = DQN
    model_name = 'dqn_v2'
    Nsteps = 2_450_000 
    Npredict = 1000

    model_filename = f"lunar_{model_name}_{Nsteps}.zip"
    model_basename = f"lunar_{model_name}_{Nsteps}"
    model_path = os.path.join(dir_path, model_filename)

    if test_many_models:
        # find models to test
        model_numbers = find_best_results(base_model_path = os.path.join(dir_path,'logs',model_basename,), min_mean = 287, max_std = 15)
        model_paths = [f"./logs/{model_basename}/dqn_check_{model_number}_steps" for model_number in model_numbers]
       # best_model_path = f'./logs/{model_basename}/best_checkpoint/best_model'
       #model_paths.append(best_model_path)

    # Create environment
    env = gym.make("LunarLander-v2",
    continuous = False,
    gravity = -10.0,
    enable_wind = False,
    wind_power = 15.0,
    turbulence_power = 1.5,)

    # params
    dqn_kwargs = dict(policy="MlpPolicy", env=env, learning_rate = 0.001, buffer_size = 75_000,\
                    learning_starts = 10_000, batch_size = 64, tau=1.0, gamma=0.99, train_freq=4, \
                        gradient_steps=1, replay_buffer_class=None, replay_buffer_kwargs=None, \
                            optimize_memory_usage=False, target_update_interval = 500, exploration_fraction=0.15,\
                                exploration_initial_eps=1.0, exploration_final_eps = 0.02,\
                                    max_grad_norm=10, stats_window_size=100, \
                                        policy_kwargs=dict(net_arch=[700, 400]), seed=0, verbose=1)
    
    model = pretrained_model(**dqn_kwargs)

    if test_many_models:
        envi = model.get_env()
        results = predict_many_models(model_paths, model_numbers, 350, envi)

    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=20)

    # Print the mean reward!
    print(f"random agent mean_reward:{mean_reward:.2f} +/- {std_reward}")

    if train_model:
        callback_on_new_best = CheckpointCallback(save_path=f'./logs/{model_basename}',  # change the path as needed
                                         name_prefix='dqn_check',
                                         verbose=1,
                                         save_freq=1000) 
        
        callback_on_eval = EvalCallback(env,
                                best_model_save_path=f'./logs/{model_basename}/best_checkpoint',  # change the path as needed
                                log_path=f'./logs/{model_basename}/log',  # change the path as needed
                                eval_freq=1000,
                                deterministic=True,
                                render=False)
        
        callback_list = CallbackList([callback_on_new_best, callback_on_eval])

        # Train the agent and display a progress bar
        model.learn(total_timesteps=int(Nsteps), callback = callback_list, progress_bar=True)

        # Save the agent
        model.save(f"{model_filename}")
        print(f"Model saved at {model_path}")


    if load_model:
        model = DQN.load(f"{model_filename}", env=env)

        mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=200)
        # Print the mean reward!
        print(f"trained agent mean_reward:{mean_reward:.2f} +/- {std_reward}")


   

if __name__ == '__main__':
    main()
