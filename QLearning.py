import numpy as np
from math import exp



def create(keys, actions) :
    n = len(keys)
    if n == 0 :
        return [0 for _ in range(actions)]
    else :
        return [create(keys[1:], actions) for _ in range(keys[0])]

def access(table, keys) :
    res = table
    for i in range(len(keys)) :
        res = res[keys[i]]
    return res

def modify(table, keys, value) :
    if len(keys) == 1 :
        table[keys[0]] = value
    else :
        modify(table[keys[0]], keys[1:], value)


def get_best_action(table, state) :
    bestAction = 0
    toStudy = access(table, state)
    bestReward = toStudy[0]
    n_actions = len(toStudy)
    
    for i in range(n_actions) :
        reward = toStudy[i]
        if reward > bestReward :
            bestReward = reward
            bestAction = i
    return bestAction


def get_best_reward(table, state) :
    bestAction = 0
    toStudy = access(table, state)
    bestReward = toStudy[0]
    n_actions = len(toStudy)
    
    for i in range(n_actions) :
        reward = toStudy[i]
        if reward > bestReward :
            bestReward = reward
            bestAction = i
    return bestReward


def train_q_learning(env, alpha = 0.1, decay_rate = 0, gamma = 0.9,
                     eps_start = 0.9, eps_end = 0.05, eps_fraction = 0.3,
                     timesteps = 2000, useProdForReward = False, maxTimestepsProd = 100) :
    
    multi_discrete = False
    n_states = 0
    try :
        n_states = [env.observation_space.n]
    except AttributeError :
        n_states = env.observation_space.nvec
        multi_discrete = True
    
    n = len(n_states)
    n_actions = env.action_space.n
    
    
    Q = create(n_states, n_actions)
    trainRewards = []
    prodRewards = []
    
    state, _ = env.reset()
    if not multi_discrete :
        state = [state]
    episode_reward = 0
    done = False
    
    for i in range(timesteps) :	# Continue until reached amount of timesteps
        if done :	# If episode ended
            if useProdForReward :
                prodRewards.append(test_q_learning(env, Q, render = False, maxTimesteps = maxTimestepsProd))   #Uses prod model to report rewards
            trainRewards.append(episode_reward)
            state, _ = env.reset()
            if not multi_discrete :
                state = [state]
            episode_reward = 0
            done = False
            
        
        # Décroissance de epsilon
        epsilon = get_new_epsilon(eps_start, eps_end, eps_fraction, (1-i/timesteps))
        
        # Décroissance de alpha
        alpha = alpha * exp(-decay_rate * i)
        
        # Choix de l'action
        if np.random.rand() < epsilon :
            action = np.random.choice(range(n_actions))
        else :
            action = get_best_action(Q, state)
            
        # On applique l'action choisie
        next_state, reward, done, _, _ = env.step(action)
        if not multi_discrete :
            next_state = [next_state]
            
        # Mise à jour de la table
        val = access(Q, state + [action])
        val += alpha * (reward + gamma * get_best_reward(Q, next_state) - access(Q, state + [action]))
        modify(Q, state + [action], val)
            
        state = next_state
        episode_reward += reward
    return Q, trainRewards, prodRewards


def test_q_learning(env, QTable, maxTimesteps = 20, render = True) :
    preventInfinite = maxTimesteps
    done = False
    state, _ = env.reset()
    multi_discrete = True
    if type(state) == int :
        multi_discrete = False
        state = [state]
    
    if render : env.render()
    total_reward = 0
    while not done and preventInfinite > 0 :
        preventInfinite -= 1
        action = get_best_action(QTable, state)
        state, reward, done, _, _ = env.step(action)
        if not multi_discrete :
            state = [state]
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

"""

class QLearningAgent() :
    
    def __init__(self) :
        pass
    
    def initialize(self, observation_space, action_space) :
        self.multi_discrete = False
        n_states = 0
        try :
            n_states = [env.observation_space.n]
        except AttributeError :
            n_states = env.observation_space.nvec
            self.multi_discrete = True
            
        self.n_actions = action_space.n
        
        self.QTable = create(n_states, self.n_actions)
        self.nbDecay = 0
    
    
    def set_parameters(self, alpha = 0.1, decay_rate = 0, gamma = 0.9,
                     eps_start = 0.9, eps_end = 0.05, eps_fraction = 0.3,
                     scheduledTimesteps = 2000) :
        self.alpha = alpha
        self.decay_rate = decay_rate
        self.gamma = gamma
        self.eps_start = eps_start
        self.epsilon = eps_start
        self.eps_end = eps_end
        self.eps_fraction = eps_fraction
        self.scheduledTimesteps = scheduledTimesteps
    
    
    def act(self, state) :
        # Choix de l'action
        if np.random.rand() < self.epsilon :
            action = np.random.choice(range(self.n_actions))
        else :
            action = get_best_action(self.QTable, state)
        return action
    
    
    def decay(self) :
        self.epsilon = get_new_epsilon(self.eps_start, self.eps_end, self.eps_fraction, (1-self.nbDecay / self.scheduledTimesteps))
        self.alpha = self.alpha * exp(-self.decay_rate * self.nbDecay)
        self.nbDecay += 1
    
    
    def learn(self, state, action, reward, next_state) :
        self.decay()
        val = access(self.QTable, state + [action])
        val += self.alpha * (reward + self.gamma * get_best_reward(self.QTable, next_state) - access(self.QTable, state + [action]))
        modify(self.QTable, state + [action], val)
"""



class QLearningAgent() :
    
    def __init__(self, env) :
        self.env = env
        self.multi_discrete = False
        n_states = 0
        try :
            n_states = [env.observation_space.n]
        except AttributeError :
            n_states = env.observation_space.nvec
            self.multi_discrete = True
        
        n = len(n_states)
        self.n_actions = env.action_space.n
        
        self.QTable = create(n_states, self.n_actions)
        self.nbDecay = 0
        
    
    def set_parameters(self, alpha = 0.1, decay_rate = 0, gamma = 0.9,
                     eps_start = 0.9, eps_end = 0.05, eps_fraction = 0.3,
                     scheduledTimesteps = 2000) :
        self.alpha = alpha
        self.decay_rate = decay_rate
        self.gamma = gamma
        self.eps_start = eps_start
        self.epsilon = eps_start
        self.eps_end = eps_end
        self.eps_fraction = eps_fraction
        self.scheduledTimesteps = scheduledTimesteps
    
    
    def act(self, state) :
        # Choix de l'action
        if np.random.rand() < self.epsilon :
            action = np.random.choice(range(self.n_actions))
        else :
            action = get_best_action(self.QTable, state)
        return action
    
    
    def train(self, timesteps = 2000, useProdForReward = False, maxTimestepsProd = 100) :
        trainRewards = []
        prodRewards = []
        
        state, _ = self.env.reset()
        if not self.multi_discrete :
            state = [state]
        episode_reward = 0
        done = False
        
        for i in range(timesteps) :	# Continue until reached amount of timesteps
            if done :	# If episode ended
                if useProdForReward :
                    prodRewards.append(self.test(render = False, maxTimesteps = maxTimestepsProd))   #Uses prod model to report rewards
                trainRewards.append(episode_reward)
                state, _ = self.env.reset()
                if not self.multi_discrete :
                    state = [state]
                episode_reward = 0
                done = False
                
            
            # On obtient l'action qu'effectue l'agent
            action = self.act(state)
                
            # On applique l'action choisie
            next_state, reward, done, _, _ = self.env.step(action, learn = True)
            if not self.multi_discrete :
                next_state = [next_state]
                
            # Mise à jour de la table
            self.learn(state, action, reward, next_state)
                
            state = next_state
            episode_reward += reward
        return self.QTable, trainRewards, prodRewards
    
    
    def decay(self) :
        self.epsilon = get_new_epsilon(self.eps_start, self.eps_end, self.eps_fraction, (1-self.nbDecay / self.scheduledTimesteps))
        self.alpha = self.alpha * exp(-self.decay_rate * self.nbDecay)
        self.nbDecay += 1
    
    
    def learn(self, state, action, reward, next_state) :
        self.decay()
        val = access(self.QTable, state + [action])
        val += self.alpha * (reward + self.gamma * get_best_reward(self.QTable, next_state) - access(self.QTable, state + [action]))
        modify(self.QTable, state + [action], val)
    
    
    def test(self, maxTimesteps = 20, render = True) :
        preventInfinite = maxTimesteps
        done = False
        state, _ = self.env.reset()
        if not self.multi_discrete :
            state = [state]
        
        if render : self.env.render()
        total_reward = 0
        while not done and preventInfinite > 0 :
            preventInfinite -= 1
            action = get_best_action(self.QTable, state)
            state, reward, done, _, _ = self.env.step(action, learn = False)
            if not self.multi_discrete :
                state = [state]
            if render : env.render()
            total_reward += reward
        if render : self.env.render()
        return total_reward
