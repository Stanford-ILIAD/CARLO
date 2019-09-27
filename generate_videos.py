"""Generate videos from state history in wandb runs."""
import pickle
import os
import gym
from moviepy.editor import ImageSequenceClip
import pandas as pd
from tensorflow import flags
import wandb
import driving_envs  # pylint: disable=unused-import

FLAGS = flags.FLAGS
flags.DEFINE_string("run_id", None, "Wandb experiment run id.")


def get_best_step(run, local_dir):
    csv_path = "ppo_driving/eval.csv"
    csv_file = run.file(csv_path)
    csv_file.download(root=local_dir, replace=True)
    eval_df = pd.read_csv(os.path.join(local_dir, csv_path), header=None)
    return int(eval_df.iloc[eval_df.iloc[:, 1].idxmax()][0])


def main():
    """Generate video from state history."""
    api = wandb.Api()
    run_path = "ayzhong/hr_adaptation/" + FLAGS.run_id
    local_dir = "/tmp/{}".format(os.path.basename(run_path))
    run = api.run(run_path)
    best_step = get_best_step(run, local_dir)
    print("Using best step: {}.".format(best_step))
    data_dicts_path = "ppo_driving/eval{}/data_dicts.pkl".format(best_step)
    data_dicts_file = run.file(data_dicts_path)
    data_dicts_file.download(root=local_dir, replace=True)
    with open(os.path.join(local_dir, data_dicts_path), "rb") as f:
        data_dicts = pickle.load(f)
    os.makedirs("vids")
    multi_env = gym.make("Merging-v0")
    for task_idx, data_dict in enumerate(data_dicts):
        for env_idx, transitions in data_dict.items():
            multi_env.reset()
            frames = []
            done = False
            for state, _action, _rew, done in transitions:
                multi_env.world.state = state
                multi_env.update_text()
                frames.append(multi_env.render(mode="rgb_array"))
            assert done, "Done should be True by end of transitions."
            clip = ImageSequenceClip(frames, fps=int(1 / multi_env.dt))
            clip.write_videofile("vids/task{}_{}.mp4".format(task_idx, env_idx))


if __name__ == "__main__":
    flags.mark_flag_as_required("run_id")
    main()
