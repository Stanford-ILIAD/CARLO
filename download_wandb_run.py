"""Download files corresponding to best ckpt in a wandb run."""
import os
from absl import app, flags
import pandas as pd
import wandb

FLAGS = flags.FLAGS
flags.DEFINE_string("run_id", None, "Wandb experiment run id.")
flags.DEFINE_string("local_path", None, "Folder to store files in locally.")

def get_best_step(run, local_dir):
    """Get the best performing checkpoint step from available run ckpts."""
    csv_path = "ppo_driving/eval.csv"
    csv_file = run.file(csv_path)
    csv_file.download(root=local_dir, replace=True)
    eval_df = pd.read_csv(os.path.join(local_dir, csv_path), header=None)
    return int(eval_df.iloc[eval_df.iloc[:, 1].idxmax()][0])


def get_last_step(run, local_dir):
    """Get the last available checkpoint step from available run ckpts."""
    csv_path = "ppo_driving/eval.csv"
    csv_file = run.file(csv_path)
    csv_file.download(root=local_dir, replace=True)
    eval_df = pd.read_csv(os.path.join(local_dir, csv_path), header=None)
    return int(eval_df[0].iloc[-1])


def download_folder(run, eval_folder, local_path):
    """Download a folder from wandb locally."""
    model_path = os.path.join(eval_folder, "model.pkl")
    obs_rms_path = os.path.join(eval_folder, "obs_rms.pkl")
    ret_rms_path = os.path.join(eval_folder, "ret_rms.pkl")
    data_dicts_path = os.path.join(eval_folder, "data_dicts.pkl")
    model_file = run.file(model_path)
    obs_rms_file = run.file(obs_rms_path)
    ret_rms_file = run.file(ret_rms_path)
    data_dicts_file = run.file(data_dicts_path)
    model_file.download(root=local_path, replace=True)
    obs_rms_file.download(root=local_path, replace=True)
    ret_rms_file.download(root=local_path, replace=True)
    data_dicts_file.download(root=local_path, replace=True)


def main(_argv):
    """Download best run checkpoint files."""
    run_id, local_path = FLAGS.run_id, FLAGS.local_path
    api = wandb.Api()
    run_path = os.path.join("ayzhong/hr_adaptation/", run_id)
    if local_path is None:
        local_path = os.path.join("/tmp", run_id)
    run = api.run(run_path)
    best_step = get_best_step(run, local_path)
    print("Downloading best step: {}.".format(best_step))
    download_folder(run, "ppo_driving/eval{}".format(best_step), local_path)
    last_step = get_last_step(run, local_path)
    print("Downloading last step: {}.".format(last_step))
    download_folder(run, "ppo_driving/eval{}".format(last_step), local_path)

if __name__ == "__main__":
    flags.mark_flag_as_required("run_id")
    app.run(main)