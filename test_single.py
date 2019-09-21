import csv
import os
import gym
from moviepy.editor import ImageSequenceClip
import numpy as np
import pandas as pd
from tensorflow import flags
import wandb
import driving_envs  # pylint: disable=unused-import

FLAGS = flags.FLAGS
flags.DEFINE_string("run_id", None, "Wandb experiment run id.")


def main():
    api = wandb.Api()
    run_path = "ayzhong/hr_adaptation/" + FLAGS.run_id
    local_dir = "/tmp/{}".format(os.path.basename(run_path))
    run = api.run(run_path)
    csv_path = "ppo_driving/eval.csv"
    csv_file = run.file(csv_path)
    csv_file.download(root=local_dir, replace=True)
    eval_df = pd.read_csv(os.path.join(local_dir, csv_path), header=None)
    best_step = int(eval_df.iloc[eval_df.iloc[:, 1].idxmax()][0])
    print("Using best step: {}.".format(best_step))
    state_path = "ppo_driving/eval{}/state_history.npy".format(best_step)
    done_path = "ppo_driving/eval{}/done_history.npy".format(best_step)
    files = run.files([state_path, done_path])
    [f.download(root=local_dir, replace=True) for f in files.objects]
    state_history = np.load(os.path.join(local_dir, state_path))
    done_history = np.load(os.path.join(local_dir, state_path))
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
