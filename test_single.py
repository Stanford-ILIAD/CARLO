import argparse
import time
import gym
from single_agent_env import make_single_env
from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv


def test(experiment_name):
    model = PPO2.load(experiment_name)
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
        time.sleep(0.04)
        i += 1
    print(ret)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_name", type=str)
    args = parser.parse_args()
    test(args.experiment_name)
