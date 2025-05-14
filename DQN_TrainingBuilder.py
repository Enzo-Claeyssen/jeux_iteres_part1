from DQN import *


class DQN_TrainingBuilder() :
    def __init__(self) :
        self.learning_rate = 0.01
        self.gamma = 0.99
        self.buffer_size = 500
        self.batch_size = 32
        self.update_freq = 500
        self.exploration_initial_eps = 1
        self.exploration_fraction = 0.3
        self.exploration_final_eps = 0.05
        
    
    def set_parameters(self, params) :
        for key, value in params :
            if key.lower() == 'learning_rate' :
                self.learning_rate = value
            elif key.lower() == 'gamma' :
                self.gamma = value
            elif key.lower() == 'buffer_size' :
                self.buffer_size = value
            elif key.lower() == 'batch_size' :
                self.batch_size = value
            elif key.lower() == 'update_freq' :
                self.update_freq = value
            elif key.lower() == 'exploration_initial_eps' :
                self.exploration_initial_eps = value
            elif key.lower() == 'exploration_fraction' :
                self.exploration_fraction = value
            elif key.lower() == 'exploration_final_eps' :
                self.exploration_final_eps = value
            else :
                raise UnknownParameterException
    
    def train(self, env, timesteps) :
        return train_dqn(env, timesteps = timesteps, useProdForReward = True,
                         learning_rate = self.learning_rate,
                         gamma = self.gamma,
                         buffer_size = self.buffer_size,
                         batch_size = self.batch_size,
                         update_freq = self.update_freq,
                         exploration_initial_eps = self.exploration_initial_eps,
                         exploration_fraction = self.exploration_fraction,
                         exploration_final_eps = self.exploration_final_eps)