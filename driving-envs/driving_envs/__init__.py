from gym.envs.registration import register

register(id="Merging-v0", entry_point="driving_envs.envs:MergingEnv")
register(id="Merging-v1", entry_point="driving_envs.envs:MergingEnv2")
register(id="Turning-v0", entry_point="driving_envs.envs:TurningEnv")
