[//]: # "Image References"

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Reinforcement Learning: Navigation

This project trains an agent to navigate (and collect bananas!) in a large, square world as in the image below. The challenge is part of the Unity environment (https://unity3d.com/machine-learning). 

![Trained Agent][image1]

### Installation

Create environment using the requirements.txt file. The code was developed in Python 3.6 and requires tensorflow==1.7.1.

1. Clone repository:

```
$ git clone https://github.com/elifons/DeepRL-Navigation.git
$ cd DeepRL-Navigation
$ pip install -r requirements.txt
```

Alternatively, follow the instractions on this link https://github.com/udacity/deep-reinforcement-learning#dependencies to set up a python environment.

2. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
3. Place the file in the DeepRL-Navigation GitHub repository, and unzip (or decompress) the file. 

### Environment

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:

- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and the environment is considered solved when the agent gets an score of +13 over 100 consecutive episodes. 

### Getting started

Example command to run the code.

```
$ python3 main.py --dest exp_dqn --n_episodes 1500
```

Or you can follow the in structions in `Navigation_notebook.ipynb` to get started with training your own agent!   

**optional arguments:**

```
  --n_episodes N_EPISODES
                        max number of training episodes (default: 500)
  --max_t MAX_T         max. number of timesteps per episode (default: 1000)
  --eps_start EPS_START
                        starting value of epsilon, for epsilon-greedy action
                        selection (default: 1.0)
  --eps_end EPS_END     minimum value of epsilon (default: 0.01)
  --eps_decay EPS_DECAY
                        multiplicative factor (per episode) for decreasing
                        epsilon (default: 0.995)
  --dest DEST           experiment dir (default: runs)
```

