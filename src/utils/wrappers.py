from collections import deque

import numpy as np
from gymnasium import Wrapper
from tensorboardX import SummaryWriter
from toymeta.dark_room import DarkRoom

from src.utils.env_batch import ParallelEnvBatch

# the following code is adapted from https://github.com/yandexdataschool/Practical_RL


class DarkRoom_with_time(DarkRoom):
    def __init__(self, episode_length, goal, *args, **kwargs):
        super().__init__(goal=goal, *args, **kwargs)
        self.time = None
        self.episode_length = episode_length

    def reset(self, *args, **kwargs):
        s, info = super().reset(*args, **kwargs)
        self.time = 1
        info["tmp"] = self.time
        return s, info

    def step(self, *args, **kwargs):
        s, r, terminated, truncated, info = super().step(*args, **kwargs)
        self.time += 1
        info["tmp"] = self.time
        if self.time == self.episode_length:
            truncated = True
        return s, r, terminated, truncated, info


class SummariesBase(Wrapper):
    """
    Env summaries writer base
    """

    def __init__(self, env, prefix=None, running_mean_size=10, step_var=None):
        super().__init__(env)
        self.episode_counter = 0
        self.prefix = prefix or self.env.spec.id
        self.step_var = step_var or 0

        self.nenvs = getattr(self.env.unwrapped, "nenvs", 1)
        self.rewards = np.zeros(self.nenvs)
        self.had_ended_episodes = np.zeros(self.nenvs, dtype=bool)
        self.episode_lengths = np.zeros(self.nenvs)
        self.reward_queues = [
            deque([], maxlen=running_mean_size) for _ in range(self.nenvs)
        ]

    def should_write_summaries(self):
        """
        Returns true if it's time to write summaries
        """
        return np.all(self.had_ended_episodes)

    def add_summaries(self):
        """
        Writes summaries
        """
        self.add_summary(
            f"Episodes/total_reward", np.mean([q[-1] for q in self.reward_queues])
        )
        self.add_summary(
            f"Episodes/reward_mean_{self.reward_queues[0].maxlen}",
            np.mean([np.mean(q) for q in self.reward_queues]),
        )
        self.add_summary(f"Episodes/episode_length", np.mean(self.episode_lengths))
        if self.had_ended_episodes.size > 1:
            self.add_summary(
                f"Episodes/min_reward", min(q[-1] for q in self.reward_queues)
            )
            self.add_summary(
                f"Episodes/max_reward", max(q[-1] for q in self.reward_queues)
            )
        self.episode_lengths.fill(0)
        self.had_ended_episodes.fill(False)

    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)
        self.rewards += rew
        self.episode_lengths[~self.had_ended_episodes] += 1

        info_collection = [info] if isinstance(info, dict) else info
        terminated_collection = (
            [terminated] if isinstance(terminated, bool) else terminated
        )
        truncated_collection = [truncated] if isinstance(truncated, bool) else truncated
        done_indices = [
            i
            for i, info in enumerate(info_collection)
            if info.get(
                "real_done", terminated_collection[i] or truncated_collection[i]
            )
        ]
        for i in done_indices:
            if not self.had_ended_episodes[i]:
                self.had_ended_episodes[i] = True
            self.reward_queues[i].append(self.rewards[i])
            self.rewards[i] = 0

        self.step_var += self.nenvs
        if self.should_write_summaries():
            self.add_summaries()
        return obs, rew, terminated, truncated, info

    def reset(self, **kwargs):
        self.rewards.fill(0)
        self.episode_lengths.fill(0)
        self.had_ended_episodes.fill(False)
        return self.env.reset(**kwargs)


class TensorboardSummaries(SummariesBase):
    """
    Writes env summaries using Tensorboard
    """

    def __init__(self, env, prefix=None, running_mean_size=10, step_var=None):
        super().__init__(env, prefix, running_mean_size, step_var)
        self.writer = SummaryWriter(f"logs/{self.prefix}")

    def add_summary(self, name, value):
        if isinstance(value, dict):
            self.writer.add_scalars(name, value, self.step_var)
        else:
            self.writer.add_scalar(name, value, self.step_var)


class _thunk:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self):
        return make_parallel_env(**self.kwargs)


def make_parallel_env(
    nenvs=None, seed=None, summaries=True, episode_length=20, goal=[0, 0], **kwargs
):
    if nenvs is not None:
        if isinstance(seed, int):
            seed = [seed] * nenvs
        if len(seed) != nenvs:
            raise ValueError("len(seed) != nenvs")

        thunks = [_thunk(**kwargs) for _ in range(nenvs)]
        env = ParallelEnvBatch(make_env=thunks, seeds=seed)

        if summaries:
            env = TensorboardSummaries(env, prefix="_".join(map(str, goal)))

        return env

    env = DarkRoom_with_time(episode_length, goal=goal, **kwargs)

    return env
