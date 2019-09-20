import os
import gym
from moviepy.editor import ImageSequenceClip
import numpy as np
from tensorflow import flags
import wandb
import driving_envs  # pylint: disable=unused-import

FLAGS = flags.FLAGS
flags.DEFINE_string("run_id", None, "Wandb experiment run id.")


def main():
    api = wandb.Api()
    run_path = "ayzhong/hr_adaptation/" + FLAGS.run_id
    run = api.run(run_path)
    rel_path = "ppo_driving/final/state_history.npy"
    local_dir = "/tmp/{}".format(os.path.basename(run_path))
    wandbfile = run.file(rel_path)
    wandbfile.download(root="/tmp/{}".format(os.path.basename(run_path)), replace=True)
    state_history = np.load(os.path.join(local_dir, rel_path))  # (T, num_envs, K)
    multi_env = gym.make("Merging-v1")
    multi_env.reset()
    frames = [multi_env.render(mode="rgb_array")]
    for state in state_history[:, 0]:
        multi_env.world.state = state
        frames.append(multi_env.render(mode="rgb_array"))
    clip = ImageSequenceClip(frames, fps=int(1 / multi_env.dt))
    clip.write_videofile(os.path.join('.', "eval.mp4"))


if __name__ == "__main__":
    flags.mark_flag_as_required("run_id")
    main()
