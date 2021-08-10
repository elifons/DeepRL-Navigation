from unityagents import UnityEnvironment
import numpy as np
from dqn_agent import Agent
from collections import deque
import torch
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os
import os.path as op

# Code based on  https://github.com/udacity/deep-reinforcement-learning.git


def dqn(dir_, n_episodes=1500, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    passed = False
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        action_size = brain.vector_action_space_size
        state = env_info.vector_observations[0]
        state_size = len(state)
        score = 0                                          # initialize the score
        for t in range(max_t):
            action = agent.act(state, eps)
            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode ha                        
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= 13.0 and passed == False:
            print('\nEnvironment reached +13 in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            passed = True
        if np.mean(scores_window)>=16.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), op.join(dir_,'checkpoint.pth'))
            break
    return scores

def create_directory(logdir):
    try:
        os.makedirs(logdir)
    except FileExistsError:
        pass



if __name__ == '__main__':  
  
    # Inputs for the main function
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n_episodes', default=500, type=int, help='max number of training episodes')
    parser.add_argument('--max_t', default=1000, type=int, help='max. number of timesteps per episode')
    parser.add_argument('--eps_start', default=1.0, type=float, help='starting value of epsilon, for epsilon-greedy action selection')
    parser.add_argument('--eps_end', default=0.01, type=float, help='minimum value of epsilon',)
    parser.add_argument('--eps_decay', default=0.995, type=float, help='multiplicative factor (per episode) for decreasing epsilon')  
    parser.add_argument('--dest', default='runs', type=str, help='experiment dir')
    args = parser.parse_args() 

    # Change the file_name parameter to match the location of the Univy environment.
    env = UnityEnvironment(file_name="Banana.app")
    path = args.dest
    create_directory(path)
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]


    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]
    # number of agents in the environment
    print('Number of agents:', len(env_info.agents))
    # number of actions
    action_size = brain.vector_action_space_size
    print('Number of actions:', action_size)
    # examine the state space 
    state = env_info.vector_observations[0]
    print('States look like:', state)
    state_size = len(state)
    print('States have length:', state_size)


    agent = Agent(state_size=state_size, action_size=action_size, seed=0)

    scores = dqn(dir_=path, n_episodes=args.n_episodes, max_t=args.max_t, eps_start=args.eps_start, eps_end=args.eps_end, eps_decay=args.eps_decay)

    env.close()


    df_scores = pd.DataFrame(scores)
    df_scores.to_csv(op.join(path, 'scores_values.csv'))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig(op.join(path, 'score.png'))
    # plt.show()

 