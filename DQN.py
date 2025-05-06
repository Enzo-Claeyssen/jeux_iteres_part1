from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback



# === CALLBACK POUR LOGGER LES RÉCOMPENSES ===
class RewardLogger(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.current_rewards = 0

    def _on_step(self):
        self.current_rewards += self.locals["rewards"][0]
        if self.locals["dones"][0]:
            self.episode_rewards.append(self.current_rewards)
            self.current_rewards = 0
        return True

# === ENTRAÎNEMENT DQN ===
def train_dqn(env, lr = 0.01, buffer_size = 500, 
              exploration_fraction = 0.3, exploration_final_eps = 0.05, 
              timesteps = 2000):
    
    env = make_vec_env(lambda: env, n_envs=1)
    logger = RewardLogger()

    model = DQN(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=lr,
        buffer_size=buffer_size,
        train_freq=1,
        target_update_interval=10,
        exploration_fraction=exploration_fraction,
        exploration_final_eps=exploration_final_eps,
    )

    model.learn(total_timesteps=timesteps, callback=logger)

    return model, logger.episode_rewards

# === TEST DU MODÈLE ===
def test_dqn(env, model, maxTimestamps = 20):
    preventInfinite = maxTimestamps
    env = make_vec_env(lambda: env, n_envs=1)
    obs = env.reset()
    trajectory = [obs]
    done = False
    while not done and preventInfinite > 0 :
        preventInfinite -= 1
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        if done :
            trajectory.append(info[0]["terminal_observation"])
        else :
            trajectory.append(obs)
        env.render("human")
        if done:
            break
    print("Trajectoire :", " → ".join(map(str, trajectory)))