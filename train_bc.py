"""Train behavior cloning model on folder of demos."""
import os
from absl import flags, app
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model

FLAGS = flags.FLAGS
flags.DEFINE_string("folder", "demos", "Folder to read data from.")
flags.DEFINE_string("save_name", "model.h5", "Output weights file name.")


def main(_argv):
    """Train and save behavior cloning model."""
    folder = FLAGS.folder
    states, actions = [], []
    for path in os.listdir(folder):
        filez = np.load(os.path.join(folder, path))
        states.append(filez["states"])
        actions.append(filez["actions"])
    states = np.concatenate(states, axis=0)
    states = np.concatenate((states[:, :6], states[:, 7:13]), axis=-1)
    actions = np.concatenate(actions, axis=0)
    h_actions = actions[:, :2]

    state_mean = np.mean(states, axis=0).tolist()
    state_std = np.std(states, axis=0).tolist()
    action_mean = np.mean(h_actions, axis=0).tolist()
    action_std = np.std(h_actions, axis=0).tolist()

    def normalize_obs(obs, mean=0, std=1):
        # HACK: re-importing tf is required for keras model loading to work.
        import tensorflow as tf  # pylint: disable=redefined-outer-name,reimported
        mean = tf.convert_to_tensor(mean, dtype=tf.float32)
        std = tf.convert_to_tensor(std, dtype=tf.float32)
        return (obs - mean) / (1e-6 + std)

    def normalize_action(act, mean=0, std=1):
        import tensorflow as tf  # pylint: disable=redefined-outer-name,reimported
        mean = tf.convert_to_tensor(mean, dtype=tf.float32)
        std = tf.convert_to_tensor(std, dtype=tf.float32)
        return act * std + mean

    model = keras.Sequential()
    model.add(
        layers.Lambda(
            normalize_obs,
            arguments={"mean": state_mean, "std": state_std},
            input_shape=(12,),
            output_shape=(12,),
        )
    )
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(2, activation=None))
    model.add(
        layers.Lambda(normalize_action, arguments={"mean": action_mean, "std": action_std})
    )
    model.compile(optimizer=keras.optimizers.Adam(0.001), loss="mse", metrics=["mse"])
    model.fit(states, h_actions, epochs=20, batch_size=32)
    model.save(FLAGS.save_name)
    # Checks that saving worked.
    del model
    model = load_model(FLAGS.save_name)
    model.evaluate(states, h_actions)


if __name__ == "__main__":
    app.run(main)
