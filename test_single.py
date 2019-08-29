import argparse
import time
import gym
from single_agent_env import make_single_env
from stable_baselines import PPO2, SAC
from stable_baselines.common.vec_env import DummyVecEnv, vec_normalize


def test(experiment_name):
    model = PPO2.load(experiment_name)
    env = vec_normalize.VecNormalize(DummyVecEnv([make_single_env]), training=False)
    env.load_running_average("{}_avgs".format(experiment_name))

    # Enjoy trained agent
    obs = env.reset()
    env.render()
    input()
    ret = 0
    i = 0
    while i < 300:
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
