import os
import torch
import numpy as np
import pandas as pd
from collections import deque
import matplotlib.pyplot as plt
from typing_extensions import Literal
from unityagents import UnityEnvironment

from Agents.PER_Agent import PER_Agent
from Agents.DQN_Agent import DQN_Agent
from Agents.DDQN_Agent import DDQN_Agent
from Agents.DuDQN_Agent import DuDQN_Agent


def init_env(env_file_name, train_mode=True):
    """
    Initial instantiation of the environment
    Params
    ======
        env_file_name: File path to the environment file
        train_mode: set the environment mode
    """
    env = UnityEnvironment(file_name=env_file_name)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=train_mode)[brain_name]
    action_size = brain.vector_action_space_size
    state_size = len(env_info.vector_observations[0])
    return env, action_size, state_size, brain_name

def get_file_ix(network:Literal['DQN', 'DDQN', 'PER', 'DuDQN']='DQN'):
    """
    Returns the latest index of files used to build specific network run.
    The files checked are: the Model Checkpoint, the training dataframe, the inference dataframe.
    Each dataframe contains episodes and the score obtained at each episode by running the relevant model checkpoint.

    Params
    ======
    network: A selection of which network model to be used
    """
    dfs = [f[:-4] for f in os.listdir('Visualizations') if f.startswith(network) and f.endswith('.csv')]
    chk = [f[:-4] for f in os.listdir('Checkpoints') if f.startswith(network) and f.endswith('.pth')]
    training_count = sum(["training" in item for item in dfs])
    inference_count = sum(["inference" in item for item in dfs])
    checkpoint_count = sum(["checkpoint" in item for item in chk])
    if training_count == inference_count == checkpoint_count:
        return training_count
    else:
        raise Exception('Files indices are incorrect')

def plot(scores, vis, network, mode, ix, params:dict):

    file_name = os.path.join(vis, network + '_' + mode + '_' + str(ix))

    df = pd.DataFrame({'episode': np.arange(len(scores)), 'score': scores})
    df['avg_score'] = df['score'].expanding().apply(lambda x: x.mean() if len(x) < 100 else x[-100:].mean(), raw=True)
    df.to_csv(file_name+'.csv')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(df['episode'], df['score'], label='score', color='b')
    plt.plot(df['episode'], df['avg_score'], label='100 episode rolling average', color='r')
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    s = 'Config\n'
    if len(params.keys())>0:
        for k, v in params.items():
            s += '{}: {}\n'.format(k,v)
    else:
        s += 'Default\n'
    ax.text(1.02, 1, s, transform=ax.transAxes, fontsize=8, verticalalignment="top", bbox=dict(facecolor="white", alpha=0.8))
    plt.legend()
    plt.title(network.upper()+' '+(mode+' '+str(ix)).title())
    plt.tight_layout()
    plt.savefig(file_name+'.png', bbox_inches='tight')
    #plt.show()
    plt.close(fig)

def compare():
    networks = ['DQN', 'DDQN', 'PER', 'DuDQN']
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    colors = ['b', 'r', 'g', 'c', 'm', 'k', 'y']
    linestyles = ['-', '--', '-.', ':', (0, (3, 5, 1, 5)), (0, (5, 10)), (0, (1, 1))]
    cnt = 0
    for network in networks:
        ix = get_file_ix(network)
        if  ix >= 1:
            for n in range(ix):
                clr = colors[cnt % len(colors)]
                ls = linestyles[(cnt // len(colors)) % len(linestyles)]
                for i, lbl in enumerate(['training','inference']):
                    filename = network+'_'+lbl+'_'+str(n)+'.csv'
                    df = pd.read_csv(os.path.join(os.curdir,'Visualizations',filename))
                    axes[i].plot(df['episode'], df['avg_score'], label=network+' '+str(n), color=clr, linestyle=ls)
                    if n == ix-1:
                        axes[i].set_ylabel('score')
                        axes[i].set_title(lbl.title())
                        axes[i].legend()
                        axes[i].grid(True)
                cnt += 1

    plt.suptitle('Network Comparison')
    plt.tight_layout()
    plt.savefig(os.path.join(os.curdir, 'Visualizations', 'Network_Comparison.png'), bbox_inches='tight')
    #plt.show()
    plt.close(fig)

def DQN_reset_env(env:UnityEnvironment, brain_name:UnityEnvironment.brain_names, train_mode:bool):
    """
    Environment Reset and Return of initial state
    Params
    ======
        env (UnityEnvironment): The interacting environment.
        brain_name (UnityEnvironment.brain_names): brain_name required for interacting with the environment
        train_mode (bool): specify whether to use the environment in train_mode or inference_mode.
    """
    env_info = env.reset(train_mode=train_mode)[brain_name]
    state = env_info.vector_observations[0]
    return state

def DQN_get_state(env:UnityEnvironment, brain_name:UnityEnvironment.brain_names, action:int):
    """
    Get next state, reward, done information from the current action
    Params
    ======
        env (UnityEnvironment): The interacting environment.
        brain_name (UnityEnvironment.brain_names): brain_name required for interacting with the environment
        action (int): Action taken to interact with the environment.
    """
    env_info = env.step(action)[brain_name]  # send the action to the environment
    next_state = env_info.vector_observations[0]  # get the next state
    reward = env_info.rewards[0]  # get the reward
    done = env_info.local_done[0]  # see if episode has finished
    return next_state, reward, done

def DQN_train(env, brain_name, agent, chk_filename, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.

    Params
    ======
        env (UnityEnvironment): The interacting environment.
        brain_name (UnityEnvironment.brain_names): brain_name required for interacting with the environment
        agent: The agent object.
        chk_filename: Path to checkpoint file for saving model params.
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    target_score = 13.0
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon

    for i_episode in range(1, n_episodes + 1):
        state = DQN_reset_env(env, brain_name,True)
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done = DQN_get_state(env, brain_name, action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon

        print('\rTraining: Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\n'.rstrip())
        if np.mean(scores_window) >= target_score:
            torch.save(agent.qnetwork_local.state_dict(), chk_filename)
            target_score = target_score + 13.0
            print('\nEnvironment solved in {:d} episodes! with Average Score: {:.2f}. Shooting for new target {}'.format(i_episode, np.mean(scores_window), target_score))
    torch.save(agent.qnetwork_local.state_dict(), chk_filename) #save last model nevertheless
    return scores

def DQN_play(env, brain_name, agent, n_episodes=2000, max_t=1000):
    """Deep Q-Learning.

    Params
    ======
        env (UnityEnvironment): The interacting environment.
        brain_name (UnityEnvironment.brain_names): brain_name required for interacting with the environment
        agent: The agent object.
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
    """
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores

    for i_episode in range(1, n_episodes + 1):
        state = DQN_reset_env(env, brain_name,True)
        score = 0
        for t in range(max_t):
            action = agent.act(state, train=False)
            next_state, reward, done = DQN_get_state(env, brain_name, action)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score

        print('\rInference Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\n'.rstrip())
    return scores

def DQN_main():

    env_file_name = os.path.join(os.path.curdir,'Banana_Windows_x86_64','Banana.exe')
    vis = os.path.join(os.path.curdir, 'Visualizations')

    #define hyperparameters for various experiments
    params = []
    params.append({'DQN': {'normalize' : False}})
    params.append({'DQN': {}})
    params.append({'DDQN': {'buffer_size': int(2e5), 'fc1': 256, 'fc2': 128}})
    params.append({'PER': {'buffer_size': int(1e6), 'fc1': 256, 'fc2': 128, 'alpha': 0.6, 'beta': 0.4}})
    params.append({'DuDQN': {'buffer_size': int(2e5), 'fc1': 256, 'fc2': 128}})

    env, action_size, state_size, brain_name = init_env(env_file_name, train_mode=True)

    net_count = {}

    for param in params:
        network = list(param.keys())[0]
        ix = get_file_ix(network)
        chk = os.path.join(os.path.curdir,'Checkpoints','{}_checkpoint_{}.pth'.format(network, ix))

        if network not in net_count:
            net_count[network] = 0
        else:
            net_count[network] += 1

        trn = os.path.join(os.path.curdir,'Visualizations','{}_training_{}.csv'.format(network, net_count[network]))
        ply = os.path.join(os.path.curdir,'Visualizations','{}_inference_{}.csv'.format(network, net_count[network]))

        param_str = 'default parameters'
        if len(param[network].keys())> 0:
            param_str = ', '.join(f'{k} = {v}' for k, v in param[network].items())

        if os.path.isfile(trn) == False:
            print(chk)
            print(trn)
            print(ply)
            if network=='DQN':
                agent = DQN_Agent(state_size=state_size, action_size=action_size, seed=0)
            if network=='DDQN':
                agent = DDQN_Agent(state_size=state_size, action_size=action_size, seed=0)
            if network=='PER':
                agent = PER_Agent(state_size=state_size, action_size=action_size, seed=0)
                state = DQN_reset_env(env, brain_name, True)
                for i in range(int(1e5)):
                    print('\rPrefilling {} Memory {}'.format(network, i), end="")
                    action = agent.act(state)
                    next_state, reward, done = DQN_get_state(env, brain_name, action)
                    state = next_state
                    if done:
                        state = DQN_reset_env(env, brain_name, True)
            if network=='DuDQN':
                agent = DuDQN_Agent(state_size=state_size, action_size=action_size, seed=0)
            agent.update(param[network])

        if network in ['DQN', 'DDQN', 'PER', 'DuDQN']:
            eps_end =  param[network]['eps_end'] if 'eps_end' in param[network].keys() else 0.01
            episode =  param[network]['episode'] if 'episode' in param[network].keys() else 2000

            if os.path.isfile(chk) == False and os.path.isfile(trn) == False:
                print('Training {} with {}'.format(network, param_str))
                scores = DQN_train(env, brain_name, agent, chk, n_episodes=episode, eps_end=eps_end)
                plot(scores, vis, network, 'training', ix, param[network])

            if os.path.isfile(chk) == True and os.path.isfile(ply) == False:
                print('Inferring from {} with {}'.format(network, param_str))
                agent.qnetwork_local.load_state_dict(torch.load(chk))
                scores = DQN_play(env, brain_name, agent, n_episodes=episode)
                plot(scores, vis, network, 'inference', ix, param[network])

    compare()

if __name__ == '__main__':
    DQN_main()

