### MDP Value Iteration 
### You must not change the arguments and output types of the given function. 
### You may debug in main and elsewhere.

import numpy as np
import gym
import time
from gym.wrappers import Monitor
from taxi_envs import *

np.set_printoptions(precision=3)


def value_iteration(env, gamma, max_iteration, tol):
    """
    Implement value iteration algorithm. You should use extract_policy to for extracting the policy.

    Parameters
    ----------
    env: OpenAI env. 
            env.P: dictionary
                    the transition probabilities of the environment
                    P[state][action] is tuples with (probability, nextstate, reward, terminal)
            env.nS: int
                    number of states
            env.nA: int
                    number of actions
    gamma: float
            Discount factor. 
    max_iteration: int
            The maximum number of iterations to run before stopping. 
    tol: float
            Determines when value function has converged.
    Returns:
    ----------
    value function: np.ndarray
    policy: np.ndarray
    """

    V = np.zeros(env.nS)
    policy = np.zeros(env.nS, dtype=int)

    ############################
    #        YOUR CODE         #
    for i in range(max_iteration):
        prev = np.copy(V)
        delta = 0
        for s in range(env.nS):
            q = [sum([env.P[s][a][0][0] * (env.P[s][a][0][2] + gamma * prev[env.P[s][a][0][1]])]) for a in range(env.nA)]
            V[s] = max(q)
            # print(np.fabs(prev_v[s] - V[s]))
            delta = max(delta, np.fabs(prev[s] - V[s]))
        if (delta <= tol):
            print ('Value-iteration converged at iteration# %d.' % (i + 1))
            break
    policy = extract_policy(env, V, gamma)
    print("Optimal Value Function:")
    print(V)
    print("Optimal Policy:")
    print(policy)
    ############################

    return V, policy






def extract_policy(env, v, gamma):
    
    """ Extract the optimal policy given the optimal value-function 
    Parameters:
    ----------
    env: OpenAI env.

    v: np.ndarray
        value function

    gamma: float
        Discount factor. Number in range [0, 1)
    Returns:
    ----------
    policy: np.ndarray
    """
    
    policy = np.zeros(env.nS, dtype=int)

    ############################
    #        YOUR CODE         #
    for s in range(env.nS):
        # q = np.zeros(env.nA)
        q = [sum([env.P[s][a][0][0] * (env.P[s][a][0][2] + gamma * v[env.P[s][a][0][1]])]) for a in range(env.nA)]
        policy[s] = np.argmax(q)
    ############################

    return policy





def example(env):
    """Run an example of the game
    Parameters
    ----------
    env: OpenAI env.
    """
    env.seed(0)
    ob = env.reset()
    for t in range(100):
        env.render()
        a = env.action_space.sample()
        ob, rew, done, _ = env.step(a)
        if done:
            break
    assert done

    env.render()


def render_episode(env, policy):
    """Run one episode for given policy on environment. 
    Parameters
    ----------
    env: OpenAI env.
        
    Policy: np.array of shape [env.nS]
        The action to take at a given state
    """

    episode_reward = 0
    ob = env.reset()
    for t in range(100):
        env.render()
        time.sleep(0.5)  
        a = policy[ob]
        ob, rew, done, _ = env.step(a)
        episode_reward += rew
        if done:
            break
    assert done
    env.render()
    print("Episode reward: %f" % episode_reward)


def avg_performance(env, policy):
    """ Evaluate the average rewards 
    Parameters
    ----------
    env: OpenAI env.
        
    Policy: np.array of shape [env.nS]
        The action to take at a given state
    """

    sum_reward = 0.
    episode = 100
    max_iteration = 6000
    for i in range(episode):
        done = False
        ob = env.reset()

        for j in range(max_iteration):
            a = policy[ob]
            ob, reward, done, _ = env.step(a)
            sum_reward += reward
            if done:
                break

    return sum_reward / i

def main():
    """
    You can test your game when you finish setting up your environment.
    Input range from 0 to 5:
        0 : South (Down)
        1 : North (Up)
        2 : East (Right)
        3 : West (Left)
        4: Pick up
        5: Drop off
    """

    GAME = "Assignment1-Taxi-v2"
    env = gym.make(GAME)
    n_state = env.observation_space.n
    print(env.P)
    print("action:")
    print(env.action_space)
    n_action = env.action_space.n
    env = Monitor(env, "taxi_simple", force=True)


    s = env.reset()
    steps = 100
    for step in range(steps):
        env.render()
        action = int(input("Please type in the next action:"))
        s, r, done, info = env.step(action)
        print(s)
        print(r)
        print(done)
        print(info)

    # close environment and monitor
    # print(env.P)
    env.close()



if __name__ == "__main__":
    
    # main()
    env = gym.make("Assignment1-Taxi-v2")
    print(env.__doc__)
    # example(env)
    V_vi, policy_vi = value_iteration(env, gamma=0.95, max_iteration=6000, tol=1e-5)
    # render_episode(env, policy_vi)
    scores = avg_performance(env, policy_vi)
    print("Average Rewards for running 100 episodes:")
    print(scores)
