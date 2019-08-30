import argparse
import os
import time
import gym
import numpy as np
from single_agent_env import make_single_env
from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv, vec_normalize


def test(experiment_name):
    model = PPO2.load(os.path.join(experiment_name, "model"))
    env = vec_normalize.VecNormalize(DummyVecEnv([make_single_env]), training=False)
    env.load_running_average(experiment_name)

    # Enjoy trained agent
    obs = env.reset()
    env.render()
    input()  # Wait until user presses key to start. Useful for video recording.
    ret = 0
    i = 0
    dones = np.array([False])
    while not np.all(dones):
        action, _states = model.predict(obs, deterministic=False)
        obs, rewards, dones, info = env.step(action)
        print(rewards)
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
