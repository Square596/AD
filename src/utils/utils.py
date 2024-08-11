import time

import numpy as np
from IPython.display import clear_output
from matplotlib import pyplot as plt
from toymeta.dark_room import DarkRoom


def visualize_policy(policy, num_steps=30):
    policy.model.eval()
    env = DarkRoom(
        size=policy.size, goal=policy.goal, terminate_on_goal=True, random_start=False
    )

    s, _ = env.reset()
    a = None
    for _ in range(num_steps):
        clear_output()
        plt.title(f"last_action = {a}")
        plt.imshow(env.render())
        plt.show()
        time.sleep(0.2)
        a = policy.act(np.array([s]))["actions"][0]
        s, r, term, trunc, _ = env.step(a)
        if term or trunc:
            break
