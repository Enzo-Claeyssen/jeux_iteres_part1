from PPO import *


class PPO_TrainingBuilder() :
    def __init__(self) :
        self.lr = 0.00003
        self.gamma = 0.99
        self.clip_range = 0.2
        self.batch_size = 64
    
    def set_parameters(self, params) :
        for key, value in params :
            if key.lower() == 'lr' :
                self.lr = value
            elif key.lower() == 'gamma' :
                self.gamma = value
            elif key.lower() == 'clip_range' :
                self.clip_range = value
            elif key.lower() == 'batch_size' :
                self.batch_size = value
            else :
                raise UnknownParameterException
    
    def train(self, env, timesteps) :
        return train_ppo(env, timesteps = timesteps, useProdForReward = True,
                         lr = self.lr,
                         gamma = self.gamma,
                         clip_range = self.clip_range,
                         batch_size = self.batch_size)