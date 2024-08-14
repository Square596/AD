import os
import subprocess
import time
from glob import glob

import numpy as np
from IPython.display import clear_output
from matplotlib import pyplot as plt
from toymeta.dark_room import DarkRoom

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
