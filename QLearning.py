import numpy as np
from math import exp


def train_q_learning(env, alpha = 0.1, gamma = 0.9, epsilon = 0.1, timesteps = 2000, useProdForReward = False) :
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))
    rewards = []
    
    state = env.reset()
    episode_reward = 0
    done = False
    
    while timesteps > 0 or not done :	# Continue until reached amount of timesteps and episode ended
        if done :	# If episode ended
            if useProdForReward :
                episode_reward = test_q_learning(env, Q, render = False)   #Uses prod model to report rewards
            rewards.append(episode_reward)
            state = env.reset()
            episode_reward = 0
            done = False
        else :
            timesteps -= 1
            
            # Choix de l'action
            if np.random.rand() < epsilon :
                action = np.random.choice(range(n_actions))
            else :
                action = np.argmax(Q[state])
            
            # On applique l'action choisie
            next_state, reward, done, _ = env.step(action)
            
            # Mise à jour de la table
            Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
            
            state = next_state
            episode_reward += reward
    return Q, rewards


def test_q_learning(env, QTable, maxTimesteps = 20, render = True) :
    preventInfinite = maxTimesteps
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