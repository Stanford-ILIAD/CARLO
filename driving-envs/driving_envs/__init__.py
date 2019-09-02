from gym.envs.registration import register

register(id="Merging-v0", entry_point="driving_envs.envs:MergingEnv", max_episode_steps=200)
register(id="Turning-v0", entry_point="driving_envs.envs:TurningEnv")
