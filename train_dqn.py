from collections import deque
import os
import random
import shutil
import time
from typing import List, Optional, Sequence, Union, NamedTuple, Tuple
from absl import app, flags
import gin
import gym
import numpy as np
import torch
from torch import from_numpy
from torch import nn
import wandb
from single_agent_env import make_single_env

FLAGS = flags.FLAGS
flags.DEFINE_multi_string("gin_file", None, "List of paths to the config files.")
flags.DEFINE_multi_string(
    "gin_param", None, "Newline separated list of Gin parameter bindings."
)


class Transition(NamedTuple):
    """Transition (or episode of transitions) tuple."""

    obs: np.ndarray
    act: Union[np.ndarray, Tuple[int, int]]
    rew: Union[np.ndarray, float]
    next_obs: np.ndarray
    done: Union[np.ndarray, bool]


class VariableLenTransitionBatch(NamedTuple):
    """Batch of episode transitions."""

    obs: Tuple[np.ndarray]
    act: Tuple[Union[np.ndarray, Tuple[int, int]]]
    rew: Tuple[Union[np.ndarray, float]]
    next_obs: Tuple[np.ndarray]
    done: Tuple[Union[np.ndarray, bool]]


@gin.configurable
class ReplayBuffer:
    """Episode-based replay buffer."""

    def __init__(self, memsize: int = 1000, max_length=None):
        self.memsize = memsize
        self.memory: Sequence[Transition] = deque(maxlen=self.memsize)
        self.max_length = max_length

    def add_episode(self, episode: Transition):
        """Add an episode to the replay buffer."""
        self.memory.append(episode)

    def get_batch(self, bsize: int) -> VariableLenTransitionBatch:
        """Get bsize episodes."""
        sampled_episodes: Sequence[Transition] = random.sample(self.memory, bsize)
        if self.max_length is not None:
            cut_episodes = []
            for episode in sampled_episodes:
                length = episode.obs.shape[0]
                if length > self.max_length:
                    point = np.random.randint(0, length + 1 - self.max_length)
                    cut_episodes.append(
                        Transition(*(d[point : point + self.max_length] for d in episode))
                    )
                else:
                    cut_episodes.append(episode)
        else:
            cut_episodes = sampled_episodes
        return VariableLenTransitionBatch(*zip(*cut_episodes))


@gin.configurable
class QNetwork(nn.Module):
    """Represents a recurrent q-function."""

    def __init__(
        self,
        input_size: Tuple[int, ...],
        out_size: int,
        rnn_hidden_size: int = 128,
        use_recurrent: bool = True,
        num_fc_layers: int = 1,
        factored: bool = False,
    ):
        super(QNetwork, self).__init__()
        self.input_size = input_size
        self.out_size = out_size
        self.rnn_hidden_size = rnn_hidden_size
        self.use_recurrent = use_recurrent
        self.factored = factored

        fc_layers = []
        prev_out_size = input_size[0]
        for _ in range(num_fc_layers):
            fc_layers.append(
                nn.Linear(in_features=prev_out_size, out_features=self.rnn_hidden_size)
            )
            prev_out_size = self.rnn_hidden_size
        self.fc_net = nn.Sequential(*fc_layers)
        self.rnn = None
        if self.use_recurrent:
            self.rnn = nn.GRU(
                input_size=self.rnn_hidden_size,
                hidden_size=self.rnn_hidden_size,
                batch_first=True,
            )
        if factored:
            bins_per_dim = int(np.sqrt(self.out_size))
            assert bins_per_dim == 11, "Should be 11 bins per dimension."
            self.adv1 = nn.Linear(in_features=self.rnn_hidden_size, out_features=bins_per_dim)
            self.adv2 = nn.Linear(in_features=self.rnn_hidden_size, out_features=bins_per_dim)
        else:
            self.adv = nn.Linear(in_features=self.rnn_hidden_size, out_features=self.out_size)
        self.val = nn.Linear(in_features=self.rnn_hidden_size, out_features=1)
        self.relu = nn.ReLU()

    def forward(
        self, obs, hidden_state=None, lengths=None
    ):  # pylint: disable=arguments-differ
        obs_shape = tuple(obs.size())  # (N, T, K)
        bsize, length = obs_shape[0], obs_shape[1]
        obs = obs.view(bsize * length, -1)  # (N * T, -1)
        features = self.fc_net(obs)  # (N * T, -1)
        rnn_hidden = None
        if self.use_recurrent:
            features = features.view(bsize, length, -1)  # (N, T, -1)
            if lengths is not None:
                features = nn.utils.rnn.pack_padded_sequence(
                    features, lengths, batch_first=True, enforce_sorted=False
                )
            # (N, T, -1), (1, N, -1)
            rnn_out, rnn_hidden = self.rnn(features, hidden_state)
            if lengths is not None:
                rnn_out, _ = nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True)
            rnn_out = torch.reshape(rnn_out, (-1, rnn_out.size()[-1]))  # (N * T, -1)
        else:
            rnn_out = features
        if self.factored:
            adv_out = self.adv1(rnn_out)[..., None] + self.adv2(rnn_out)[..., None, :]
            adv_out = adv_out.view(bsize * length, -1)
        else:
            adv_out = self.adv(rnn_out)  # (N * T, num_actions)
        val_out = self.val(rnn_out)  # (N * T, 1)
        q_out = adv_out + val_out  # (N * T, num_actions)
        q_out = adv_out
        q_out = q_out.view(bsize, length, -1)
        return q_out, rnn_hidden


class Policy:
    "Abstract policy class."

    def act(self, obs: np.ndarray, progress: Optional[float] = None) -> Tuple[int, int]:
        """Returns an action from observation."""
        raise NotImplementedError("Implement in subclass.")

    def reset(self):
        """Called between episodes."""
        raise NotImplementedError("Implement in subclass.")


class RandomPolicy:
    def act(self, obs, progress=None):
        del obs, progress
        return np.random.randint(121)

    def reset(self):
        pass


log_softmax = nn.LogSoftmax(dim=-1)
mse = nn.MSELoss()


class TdError(nn.Module):
    """TD error module."""

    def __init__(self, q_network, q_target_network, gamma):
        super(TdError, self).__init__()
        self.gamma = gamma
        self.q_network = q_network
        self.q_target_network = q_target_network

    def forward(self, transition_th, lengths):
        obs_th, act_th, rew_th, next_obs_th, done_th = transition_th
        with torch.no_grad():
            next_q_values, _ = self.q_target_network(next_obs_th, lengths=lengths)
            next_values, _ = next_q_values.max(dim=-1)
            target = rew_th + self.gamma * (1.0 - done_th) * next_values
        q_values, _ = self.q_network(obs_th, lengths=lengths)
        Q_s = q_values.gather(dim=-1, index=act_th.unsqueeze(-1)).squeeze()
        return mse(Q_s, target)


def collect_episode(
    env: gym.Env, policy: Policy, progress: Optional[float] = None
) -> Transition:
    """Collect an env episode using policy."""
    done = False
    policy.reset()
    obs = env.reset()
    episode_data: List[Transition] = []
    while not done:
        act = policy.act(obs, progress=progress)
        next_obs, rew, done, debug = env.step(act)
        del debug
        episode_data.append(Transition(obs, act, rew, next_obs, done))
        obs = next_obs
    transitions = Transition(*[np.stack(x) for x in zip(*episode_data)])
    return transitions


@gin.configurable
def compute_epsilon(
    training_progress: float,
    initial_epsilon: float = 1.0,
    final_epsilon: float = 0.0,
    explore_time_ratio: float = 0.1,
) -> float:
    """Calculates the current epsilon according to linear decay schedule."""
    m = (final_epsilon - initial_epsilon) / explore_time_ratio
    return max(m * training_progress + initial_epsilon, final_epsilon)


class TorchPolicy(Policy):
    """Policy based on torch network."""

    def __init__(self, q_net: QNetwork, device):
        self.q_net = q_net
        self.device = device
        self.hidden_state = None

    def act(self, obs: np.ndarray, progress: Optional[float] = None) -> int:
        obs_torch = torch.from_numpy(obs).float().to(self.device)
        obs_torch = obs_torch.unsqueeze(0).unsqueeze(0)
        act_torch, self.hidden_state = self.q_net(obs_torch, self.hidden_state)
        act = int(np.argmax(act_torch.squeeze().detach().cpu().numpy()))
        # Act randomly with probability eps (independently).
        if progress is not None and np.random.rand() < compute_epsilon(progress):
            act = np.random.randint(121)
        return act

    def reset(self):
        self.hidden_state = None


def make_fixed_length(arrays, lengths):
    max_length = max(lengths)
    batch_size = len(arrays)
    padded_data = np.zeros((batch_size, max_length) + arrays[0].shape[1:])
    for i, length in enumerate(lengths):
        padded_data[i, :length] = arrays[i]
    return torch.from_numpy(padded_data)


def transition_to_torch(transition_batch: VariableLenTransitionBatch, device):
    lengths = [arr.shape[0] for arr in transition_batch.obs]
    obs = make_fixed_length(transition_batch.obs, lengths).float().to(device)
    act = make_fixed_length(transition_batch.act, lengths).long().to(device)
    rew = make_fixed_length(transition_batch.rew, lengths).float().to(device)
    next_obs = make_fixed_length(transition_batch.next_obs, lengths).float().to(device)
    done = make_fixed_length(transition_batch.done, lengths).float().to(device)
    data = [obs, act, rew, next_obs, done]
    return data, torch.Tensor(lengths).long()


def step(sampled_batch: VariableLenTransitionBatch, error_module, optimizer, device):
    """Take a single gradient step on sampled_batch."""
    torch_batch, lengths = transition_to_torch(sampled_batch, device)
    optimizer.zero_grad()
    td_error = error_module(torch_batch, lengths)
    optimizer.zero_grad()
    td_error.backward()
    optimizer.step()
    return td_error.detach().cpu().numpy()


def eval_policy(env: gym.Env, policy: Policy, eval_episodes: int):
    """Evaluate policies."""
    eval_data = []
    for _ in range(eval_episodes):
        episode_data = collect_episode(env, policy)
        eval_data.append(episode_data)
    avg_eval_ret = np.mean([np.sum(t.rew) for t in eval_data])
    return {"eval_return": avg_eval_ret}


@gin.configurable
def train(
    save_freq: int = 200,
    num_training_iterations: int = 5000,
    init_collect_eps: int = 100,
    learning_rate: float = 0.00025,
    train_freq: int = 1,
    train_batch_size: int = 32,
    target_update_freq: int = 50,
    num_grad_steps_per_train: int = 5,
    eval_freq: int = 50,
    eval_episodes: int = 1,
    gamma: float = 0.995,
):
    if os.path.exists("dqn_out"):
        shutil.rmtree("dqn_out")
    os.makedirs("dqn_out")
    wandb.init(project="hr-adaptation")
    env = make_single_env(discrete=True)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    q_net = QNetwork(env.observation_space.shape, env.action_space.n).to(device)
    q_target_net = QNetwork(env.observation_space.shape, env.action_space.n).to(device)
    td_error_module = TdError(q_net, q_target_net, gamma).to(device)
    wandb.watch(td_error_module)
    optimizer = torch.optim.Adam(q_net.parameters(), lr=learning_rate)
    q_target_net.load_state_dict(q_net.state_dict())
    random_policy = RandomPolicy()
    policy = TorchPolicy(q_net, device)
    rb = ReplayBuffer()
    with open("dqn_out/operative_config.gin", "w") as f:
        f.write(gin.operative_config_str())
        wandb.save("dqn_out/operative_config.gin")
    for i in range(init_collect_eps):
        rb.add_episode(collect_episode(env, random_policy))
    last_time = time.time()
    log_dict = {}
    for i in range(num_training_iterations):
        ep_transition = collect_episode(env, policy, progress=i / num_training_iterations)
        ret = np.sum(ep_transition.rew)
        log_dict.update({"collect_return": ret})
        rb.add_episode(ep_transition)
        if i % target_update_freq == 0:
            q_target_net.load_state_dict(q_net.state_dict())
        if i % train_freq == 0:
            td_errors = []
            for _ in range(num_grad_steps_per_train):
                sampled_batch = rb.get_batch(train_batch_size)
                td_errors.append(step(sampled_batch, td_error_module, optimizer, device))
            log_dict.update({"td_error": np.mean(td_errors)})
        if i > 0 and i % eval_freq == 0:
            log_dict.update(eval_policy(env, policy, eval_episodes))
            steps_per_sec = eval_freq / (time.time() - last_time)
            last_time = time.time()
            log_dict.update({"steps_per_sec": steps_per_sec})
            print("{:d}:\tSteps/sec:\t{:.1f}".format(i, steps_per_sec))
        if i % save_freq == 0:
            filename = "dqn_out/model{}".format(i)
            torch.save(q_net.state_dict(), filename)
            wandb.save(filename)
        wandb.log(log_dict, step=i)


def main(_):
    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)
    train()


if __name__ == "__main__":
    app.run(main)
