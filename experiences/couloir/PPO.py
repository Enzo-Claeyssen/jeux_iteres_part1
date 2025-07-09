from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback


# === CALLBACK POUR LOGGER LES RÉCOMPENSES ===
class RewardLoggerPPO(BaseCallback):
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
                self.prod_episode_rewards.append(test_ppo(self.env, self.locals['self'], maxTimesteps = 100, render = False))
            self.train_episode_rewards.append(self.current_rewards)
            self.current_rewards = 0
        return True

# === ENTRAÎNEMENT PPO ===
def train_ppo(env,
              timesteps = 2000, useProdForReward = False, lr = 0.00003, gamma = 0.99, clip_range = 0.2, batch_size = 64):
    
    logger = RewardLoggerPPO(env, useProdForReward = useProdForReward)
    env = make_vec_env(lambda: env, n_envs=1)

    model = PPO(
        "MlpPolicy",
        env,
        verbose = 0,
        learning_rate = lr,
        gamma = gamma,
        clip_range = clip_range,
        batch_size = batch_size
    )

    model.learn(total_timesteps=timesteps, callback=logger)

    return model, logger.train_episode_rewards, logger.prod_episode_rewards

# === TEST DU MODÈLE ===
def test_ppo(env, model, maxTimesteps = 20, render = True):
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
            env.render()
        
        if done:
            break
    
    if render :
        print("Trajectoire :", " → ".join(map(str, trajectory)))
    return total_reward