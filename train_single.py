import time
import gym
from single_agent_env import PidSingleEnv, PidDiscreteSingleEnv
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy, MlpLnLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv


def make_single_env():
    multi_env = gym.make("Merging-v0")
    env = PidSingleEnv(multi_env)
    return env


def train():
    env = DummyVecEnv([make_single_env])
    model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log='/tmp/ppo_driving', n_steps=400)
    model.learn(total_timesteps=500000)
    model.save("ppo_driving")


def test():
    model = PPO2.load("ppo_driving")
    env = DummyVecEnv([make_single_env])

    # Enjoy trained agent
    obs = env.reset()
    env.render()
    ret = 0
    i = 0
    while i < 400:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        ret += rewards
        env.render()
        time.sleep(.04)
        i += 1
    print(ret)


if __name__ == "__main__":
    test()
