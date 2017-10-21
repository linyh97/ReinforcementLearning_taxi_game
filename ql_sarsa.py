### Model free learning using Q-learning and SARSA
### You must not change the arguments and output types of the given function. 
### You may debug in Main and elsewhere.

import numpy as np
import gym
import time
from gym.wrappers import Monitor
from taxi_envs import *
import pylab as pl
from random import *
import random as rand

def QLearning(env, num_episodes, gamma, lr, e):
    """Implement the Q-learning algorithm following the epsilon-greedy exploration.
    Update Q at the end of every episode.

    Parameters
    ----------
    env: gym.core.Environment
      Environment to compute Q function
    num_episodes: int 
      Number of episodes of training.
    gamma: float
      Discount factor. 
    learning_rate: float
      Learning rate. 
    e: float
      Epsilon value used in the epsilon-greedy method. 


    Returns
    -------
    np.array
      An array of shape [env.nS x env.nA] representing state, action values
    """

    ############################
    #         YOUR CODE        #
    ############################

    Q = np.zeros((env.nS, env.nA))
    step_count = [0 for x in range(num_episodes)]
    total_count = 0
    sum_reward = 0
    avg_reward = [0 for x in range(num_episodes)]
    total_reward = [0 for x in range(num_episodes)]
    for i in range(num_episodes):
        s = env.reset()
        done = False
        while not done:
            step_count[i] += 1
            total_count += 1
            if rand.random() > e:
                a = np.argmax(Q[s])
            else:
                a = rand.randint(0, 5)
            sp, r, done, _ = env.step(a)
            sum_reward += r
            total_reward[i] += r
            if not done:
                Q[s, a] += lr * (r + gamma * np.max(Q[sp]) - Q[s, a])
            else:
                Q[s, a] += lr * (r - Q[s, a])
            s = sp
        if i > 0:
            avg_reward[i] += sum_reward / i

    pl.plot(range(num_episodes), avg_reward)
    pl.title('QLearning: learning progress for training 1000 episodes')
    pl.xlabel('Episode numbers')
    pl.ylabel('Average rewards')
    pl.show()
    pl.plot(range(num_episodes), step_count)
    pl.title('QLearning: Episodes length for training 1000 episodes')
    pl.xlabel('Episode numbers')
    pl.ylabel('Number of steps')
    pl.show()
    print("Q_Learning's Q value:")
    print(Q)
    return Q


def SARSA(env, num_episodes, gamma, lr, e):
    """Implement the SARSA algorithm following epsilon-greedy exploration.
    Update Q at the end of every episode.

    Parameters
    ----------
    env: gym.core.Environment
      Environment to compute Q function 
    num_episodes: int 
      Number of episodes of training
    gamma: float
      Discount factor. 
    learning_rate: float
      Learning rate. 
    e: float
      Epsilon value used in the epsilon-greedy method. 


    Returns
    -------
    np.array
      An array of shape [env.nS x env.nA] representing state-action values
    """

    ############################
    #         YOUR CODE        #
    step_count = [1 for x in range(num_episodes)]
    total_count = 1
    sum_reward = 0
    avg_reward = [0 for x in range(num_episodes)]
    total_reward = [0 for x in range(num_episodes)]
    Q = np.ones((env.nS, env.nA))
    for i in range(num_episodes):
        s = env.reset()
        done = False
        if rand.random() > e:
            a = np.argmax(Q[s])
        else:
            a = rand.randint(0, 5)
            s, r, done1, _ = env.step(a)
            sum_reward += r
        while not done:
            step_count[i] += 1
            total_count += 1
            sp, r, done, _ = env.step(a)
            sum_reward += r
            if rand.random() > e:
                ap = np.argmax(Q[sp])
            else:
                ap = rand.randint(0, 5)
            Q[s, a] += lr * (r + gamma * Q[sp, ap] - Q[s, a])
            s = sp
            a = ap
        if i > 0:
            avg_reward[i] += sum_reward / i

    # pl.plot(range(num_episodes), avg_reward)
    # pl.title('SARSA: learning progress for training 1000 episodes')
    # pl.xlabel('Episode numbers')
    # pl.ylabel('Average rewards')
    # pl.show()
    # pl.plot(range(num_episodes), step_count)
    # pl.title('SARSA: Episodes length for training 1000 episodes')
    # pl.xlabel('Episode numbers')
    # pl.ylabel('Number of steps')
    # pl.show()

    print("SARSA's Q value:")
    print(Q)
    return Q
    ############################



def render_episode_Q(env, Q):
    """Renders one episode for Q functionon environment.

      Parameters
      ----------
      env: gym.core.Environment
        Environment to play Q function on. 
      Q: np.array of shape [env.nS x env.nA]
        state-action values.
    """

    episode_reward = 0
    state = env.reset()
    done = False
    while not done:
        env.render()
        time.sleep(0.5)  
        action = np.argmax(Q[state])
        state, reward, done, _ = env.step(action)
        episode_reward += reward

    print ("Episode reward: %f" %episode_reward)



def main():
    env = gym.make("Assignment1-Taxi-v2")
    Q_QL = QLearning(env, num_episodes=1000, gamma=0.95, lr=0.1, e=0.1)
    Q_Sarsa = SARSA(env, num_episodes=1000, gamma=0.95, lr=0.1, e=0.1)
    #render_episode_Q(env, Q_QL)



if __name__ == '__main__':
    main()
