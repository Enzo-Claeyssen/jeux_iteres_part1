import numpy as np
from math import exp


def train_q_learning(env, alpha = 0.1, gamma = 0.9,
                     eps_start = 0.9, eps_end = 0.05, eps_fraction = 0.3,
                     timesteps = 2000, useProdForReward = False, maxTimestepsProd = 100) :
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))
    trainRewards = []
    prodRewards = []
    
    state = env.reset()
    episode_reward = 0
    done = False
    
    for i in range(timesteps) :	# Continue until reached amount of timesteps
        if done :	# If episode ended
            if useProdForReward :
                prodRewards.append(test_q_learning(env, Q, render = False, maxTimesteps = maxTimestepsProd))   #Uses prod model to report rewards
            trainRewards.append(episode_reward)
            state = env.reset()
            episode_reward = 0
            done = False
            
        
        # Décroissance de epsilon
        epsilon = get_new_epsilon(eps_start, eps_end, eps_fraction, (1-i/timesteps))
        
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
    return Q, trainRewards, prodRewards


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


def get_new_epsilon(start, end, fraction, progress_remaining) :
    """
    start : starting epsilon value
    end : last epsilon value
    fraction : if 0.1, then after 10% of training completed epsilon should be equal to end
    progress_remaing : % / 100 of progress remaining (0 if ended, 1 if starting)
    """
    if (1 - progress_remaining) > fraction :
        return end
    else :
        return start + (1 - progress_remaining) * (end - start) / fraction