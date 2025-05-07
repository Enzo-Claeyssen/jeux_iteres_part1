from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback



# === CALLBACK POUR LOGGER LES RÉCOMPENSES ===
class RewardLogger(BaseCallback):
    def __init__(self, env, verbose=0, useProdForReward = False):
        super().__init__(verbose)
        self.env = env
        self.train_episode_rewards = []
        self.prod_episode_rewards = []
        self.current_rewards = 0
        self.useProdForReward = useProdForReward

    def _on_step(self):
        self.current_rewards += self.locals["rewards"][0]
        if self.locals["dones"][0]:
            if self.useProdForReward :
                self.prod_episode_rewards.append(test_dqn(self.env, self.locals['self'], maxTimesteps = 100, render = False))
            self.train_episode_rewards.append(self.current_rewards)
            self.current_rewards = 0
        return True

# === ENTRAÎNEMENT DQN ===
def train_dqn(env, lr = 0.01, gamma = 0.99, buffer_size = 500, update_freq = 500,
              exploration_initial_eps = 1, exploration_fraction = 0.3, exploration_final_eps = 0.05, 
              timesteps = 2000, useProdForReward = False):
    
    logger = RewardLogger(env, useProdForReward = useProdForReward)
    env = make_vec_env(lambda: env, n_envs=1)

    model = DQN(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=lr,
        gamma = gamma,
        buffer_size=buffer_size,
        train_freq=1,
        target_update_interval=update_freq,
        exploration_initial_eps = exploration_initial_eps,
        exploration_fraction=exploration_fraction,
        exploration_final_eps=exploration_final_eps,
    )

    model.learn(total_timesteps=timesteps, callback=logger)

    return model, logger.train_episode_rewards, logger.prod_episode_rewards

# === TEST DU MODÈLE ===
def test_dqn(env, model, maxTimesteps = 20, render = True):
    preventInfinite = maxTimesteps
    env = make_vec_env(lambda: env, n_envs=1)
    obs = env.reset()
    trajectory = [obs]
    done = False
    total_reward = 0
    
    while not done and preventInfinite > 0 :
        preventInfinite -= 1
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if done :
            trajectory.append(info[0]["terminal_observation"])
        else :
            trajectory.append(obs)
        
        if render :
            env.render("human")
        
        if done:
            break
    
    if render :
        print("Trajectoire :", " → ".join(map(str, trajectory)))
    return total_reward