from gym.envs.registration import register

register(id="Merging-v0", entry_point="driving_envs.envs:MergingEnv")
register(id="Turning-v0", entry_point="driving_envs.envs:TurningEnv")
