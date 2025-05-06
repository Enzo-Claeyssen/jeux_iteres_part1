import numpy as np
from math import exp

def train_q_learning(env, alpha = 0.1, gamma = 0.9, epsilon = 0.1, episodes = 100) :
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))
    rewards = []
    
    for episode in range(episodes) :
        state = env.reset()
        total_reward = 0
        done = False
        while not done :
            if np.random.rand() < epsilon :
                action = np.random.choice(range(n_actions))
            else :
                action = np.argmax(Q[state])
            next_state, reward, done, _ = env.step(action)
            Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
            state = next_state
            total_reward += reward
        # total_reward = test_q_learning(env, Q, render = False)   Use prod model to report rewards
        rewards.append(total_reward)
    
    return Q, rewards

def test_q_learning(env, QTable, maxTimestamps = 20, render = True) :
    preventInfinite = maxTimestamps
    done = False
    state = env.reset()
    if render : env.render()
    total_reward = 0
    while not done and preventInfinite > 0 :
        preventInfinite -= 1
        action = np.argmax(QTable[state])
        state, reward, done, _ = env.step(action)
        if render : env.render()
        total_reward += reward
    return total_reward