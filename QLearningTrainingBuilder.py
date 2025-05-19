from QLearning import *


class QLearningTrainingBuilder() :
    def __init__(self) :
        self.alpha = 0.1
        self.decay_rate = 0
        self.gamma = 0.9
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_fraction = 0.3
    
    def set_parameters(self, params) :
        for key, value in params :
            if key.lower() == 'alpha' :
                self.alpha = value
            elif key.lower() == 'decay_rate' :
                self.decay_rate = value
            elif key.lower() == 'gamma' :
                self.gamma = value
            elif key.lower() == 'eps_start' :
                self.eps_start = value
            elif key.lower() == 'eps_end' :
                self.eps_end = value
            elif key.lower() == 'eps_fraction' :
                self.eps_fraction = value
            else :
                raise UnknownParameterException
    
    def train(self, env, timesteps) :
        return train_q_learning(env, timesteps = timesteps, useProdForReward = True,
                                alpha = self.alpha,
                                decay_rate = self.decay_rate,
                                gamma = self.gamma,
                                eps_start = self.eps_start,
                                eps_end = self.eps_end,
                                eps_fraction = self.eps_fraction)