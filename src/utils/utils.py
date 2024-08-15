import os
import subprocess
import time
from collections import deque
from glob import glob

import numpy as np
import torch
from IPython.display import clear_output
from matplotlib import pyplot as plt
from torch.distributions import Categorical
from toymeta.dark_room import DarkRoom

from src.utils.wrappers import DarkRoom_with_time

plt.ioff()


def visualize_policy(policy, greedy=True, num_steps=30):
    policy.model.eval()
    env = DarkRoom(
        size=policy.size, goal=policy.goal, terminate_on_goal=True, random_start=False
    )

    goal_name = "_".join(map(str, policy.goal))
    if not os.path.exists("src/videos"):
        os.mkdir("src/videos")

    s, _ = env.reset()
    a = None
    for i in range(num_steps):
        plt.title(f"last_action = {a}")
        plt.imshow(env.render())
        plt.savefig(f"src/videos/{goal_name}__%02d.png" % i)
        plt.close()

        if greedy:
            a = int(policy.model(np.array([s]))[1].argmax())
        else:
            a = policy.act(np.array([s]))["actions"][0]

        s, r, term, trunc, _ = env.step(a)
        if term or trunc:
            break

    plt.title(f"last_action = {a}")
    plt.imshow(env.render())
    plt.savefig(f"src/videos/{goal_name}__%02d.png" % (i + 1))
    plt.close()

    subprocess.call(
        [
            "ffmpeg",
            "-framerate",
            "8",
            "-i",
            f"src/videos/{goal_name}__%02d.png",
            "-r",
            "30",
            "-pix_fmt",
            "yuv420p",
            f"src/videos/{goal_name}.mp4",
        ]
    )

    for file_name in glob("src/videos/*.png"):
        os.remove(file_name)

    clear_output()
    env.close()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def eval_goal(
    dt,
    length,
    time_rel,
    goal,
    episode_length,
    mean_ep,
    writer,
    size,
    terminate_on_goal=True,
    random_start=False,
):
    dt.eval()
    env = DarkRoom_with_time(
        episode_length=episode_length,
        goal=goal,
        size=size,
        terminate_on_goal=terminate_on_goal,
        random_start=random_start,
    )
    s, info = env.reset()

    states = torch.zeros(1, length + 1, dtype=torch.int)
    states[:, 0] = torch.as_tensor(s)

    actions = torch.zeros(1, length, dtype=torch.int)

    returns = torch.zeros(1, length + 1, dtype=torch.int)

    if time_rel:
        time_steps = torch.zeros(1, length + 1, dtype=torch.int)
        time_steps[:, 0] = torch.as_tensor(info["tmp"])
    else:
        time_steps = torch.arange(length, dtype=torch.int)
        time_steps = time_steps.view(1, -1)

    ep_returns_queue = deque([], maxlen=mean_ep)
    ep_len_queue = deque([], maxlen=mean_ep)

    ep_return, ep_len = 0.0, 0.0
    for step in range(length):
        predicted_logits = dt(
            states=states[:, : step + 1],
            actions=actions[:, : step + 1],
            returns=returns[:, : step + 1],
            time_steps=time_steps[:, : step + 1],
        )

        last_logit = predicted_logits[0, -1, :]
        distribution = Categorical(logits=last_logit)
        action = int(distribution.sample())

        next_s, reward, done, trunc, info = env.step(action)

        states[:, step + 1] = torch.as_tensor(next_s)
        actions[:, step] = torch.as_tensor(action)
        returns[: step + 1] = torch.as_tensor(reward)
        time_steps[: step + 1] = torch.as_tensor(info["tmp"])

        ep_return += reward
        ep_len += 1

        if done or trunc:
            ep_returns_queue.append(ep_return)
            ep_len_queue.append(ep_len)

            writer.add_scalar(
                f"eval/reward_mean_{mean_ep}", np.mean(ep_returns_queue), step
            )
            writer.add_scalar(
                f"eval/episode_length_mean_{mean_ep}", np.mean(ep_len_queue), step
            )
            ep_return, ep_len = 0.0, 0.0
            
            next_s, info = env.reset()
            states[:, step + 1] = torch.as_tensor(next_s)
            time_steps[: step + 1] = torch.as_tensor(info["tmp"])
