"""Generate videos from data dicts file."""
import pickle
import os
import gym
from moviepy.editor import ImageSequenceClip
from tensorflow import flags
import driving_envs  # pylint: disable=unused-import

FLAGS = flags.FLAGS
flags.DEFINE_string("data_dict", None, "Path to data dict.")


def main():
    """Generate video from data dicts file."""
    with open(FLAGS.data_dict, "rb") as f:
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
    flags.mark_flag_as_required("data_dict")
    main()
