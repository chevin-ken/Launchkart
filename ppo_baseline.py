import os
import gym
import gym_mupen64plus

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True

CHECKPOINT_DIR = '/src/launchkart/train/'
LOG_DIR = '/src/launchkart/logs/'

callback = TrainAndLoggingCallback(check_freq=1, save_path=CHECKPOINT_DIR)

env = gym.make('Mario-Kart-Discrete-Luigi-Raceway-v0')

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log = LOG_DIR)
model.learn(total_timesteps=1)
print("Done learning")

model = PPO.load("/src/launchkart/train/best_model_1")

obs = env.reset()

for i in range(10000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    print(rewards)