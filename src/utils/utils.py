import glob
import os
import subprocess
import time

import numpy as np
from IPython.display import clear_output
from matplotlib import pyplot as plt
from toymeta.dark_room import DarkRoom

plt.ioff()


def visualize_policy(policy, num_steps=30):
    policy.model.eval()
    env = DarkRoom(
        size=policy.size, goal=policy.goal, terminate_on_goal=True, random_start=False
    )

    s, _ = env.reset()
    a = None
    for i in range(num_steps):
        plt.title(f"last_action = {a}")
        plt.imshow(env.render())
        plt.savefig("videos/img%02d.png" % i)
        plt.close()

        a = int(policy.model(np.array([s]))[1].argmax())
        s, r, term, trunc, _ = env.step(a)
        if term or trunc:
            os.chdir("videos")
            subprocess.call(
                [
                    "ffmpeg",
                    "-framerate",
                    "8",
                    "-i",
                    "img%02d.png",
                    "-r",
                    "30",
                    "-pix_fmt",
                    "yuv420p",
                    f'{"_".join(map(str, policy.goal))}.mp4',
                ]
            )

            for file_name in glob.glob("*.png"):
                os.remove(file_name)

            clear_output()
            break
