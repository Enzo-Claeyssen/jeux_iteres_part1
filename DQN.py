from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
import torch as th



# === CALLBACK POUR LOGGER LES RÉCOMPENSES ===
class RewardLogger(BaseCallback):
    def __init__(self, env, verbose=0, useProdForReward = False, maxTimestepsProd = 100):
        super().__init__(verbose)
        self.env = env
        self.train_episode_rewards = []
        self.prod_episode_rewards = []
        self.current_rewards = 0
        self.useProdForReward = useProdForReward
        self.maxTimestepsProd=  maxTimestepsProd

    def _on_step(self):
        self.current_rewards += self.locals["rewards"][0]
        if self.locals["dones"][0]:
            if self.useProdForReward :
                self.prod_episode_rewards.append(test_dqn(self.env, self.locals['self'], maxTimesteps = self.maxTimestepsProd, render = False))
            self.train_episode_rewards.append(self.current_rewards)
            self.current_rewards = 0
        return True

# === ENTRAÎNEMENT DQN ===
def train_dqn(env, learning_rate = 0.01, gamma = 0.99, buffer_size = 500, batch_size = 32, update_freq = 500, net_arch = [64, 64],
              exploration_initial_eps = 1, exploration_fraction = 0.3, exploration_final_eps = 0.05,
              timesteps = 2000, useProdForReward = False, maxTimestepsProd = 100, use_ReLU = True):
    
    logger = RewardLogger(env, useProdForReward = useProdForReward, maxTimestepsProd = maxTimestepsProd)
    env = make_vec_env(lambda: env, n_envs=1)

    if use_ReLU :
        policy_kwargs = dict(activation_fn=th.nn.ReLU,
                             net_arch=net_arch)
    else :
        policy_kwargs = dict(activation_fn = th.nn.Sigmoid,
                             net_arch=net_arch)

    model = DQN(
        "MlpPolicy",
        env,
        verbose=0,
        policy_kwargs = policy_kwargs,
        learning_rate=learning_rate,
        gamma = gamma,
        buffer_size=buffer_size,
        batch_size = batch_size,
        train_freq=1,
        target_update_interval=update_freq,
        exploration_initial_eps = exploration_initial_eps,
        exploration_fraction=exploration_fraction,
        exploration_final_eps=exploration_final_eps
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