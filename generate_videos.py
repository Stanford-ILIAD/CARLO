"""Generate videos from state history in wandb runs."""
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
    """Generate video from state history."""
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
    state_file, done_file = run.file(state_path), run.file(done_path)
    state_file.download(root=local_dir, replace=True)
    done_file.download(root=local_dir, replace=True)
    state_history = np.load(os.path.join(local_dir, state_path))
    done_history = np.load(os.path.join(local_dir, done_path))
    os.makedirs("vids")
    multi_env = gym.make("Merging-v0")
    for i in range(state_history.shape[1]):
        multi_env.reset()
        frames = []
        episode_number = 0
        for j, state in enumerate(state_history[:, i, :]):
            is_done = done_history[j, i]
            multi_env.world.state = state
            multi_env.update_text()
            frames.append(multi_env.render(mode="rgb_array"))
            if is_done:
                clip = ImageSequenceClip(frames, fps=int(1 / multi_env.dt))
                clip.write_videofile("vids/eval{}_{}.mp4".format(i, episode_number))
                multi_env.reset()
                frames = []
                episode_number += 1


if __name__ == "__main__":
    flags.mark_flag_as_required("run_id")
    main()
