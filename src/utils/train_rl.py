import os
import numpy as np
import torch
from tensorboardX import SummaryWriter
from tqdm import trange

from src.utils.custom_transforms import ComputeValueTargets, MergeTimeBatch
from utils.a2c import A2C, MLP_model, Policy
from src.utils.runners import EnvRunner
from src.utils.utils import visualize_policy
from src.utils.wrappers import make_parallel_env


def train_A2C(goal):
    GOAL = goal
    NENVS = 10
    SEED = 911

    SIZE = 9
    TERMINATE_ON_GOAL = True
    RANDOM_START = False

    HIDDEN_SIZE = 128
    N_ACTIONS = 5

    NSTEPS = 10

    ENV_STEPS = 2_500_000
    NUM_EPOCH = int(ENV_STEPS / NENVS / NSTEPS)
    LR = 1e-4
    BETA1 = 0.9
    BETA2 = 0.999
    EPS = 1e-6
    VALUE_LOSS_COEF = 0.99
    ENTROPY_COEF = 0.01

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    goal_name = "_".join(map(str, GOAL))

    env = make_parallel_env(
        nenvs=NENVS,
        seed=SEED,
        size=SIZE,
        terminate_on_goal=TERMINATE_ON_GOAL,
        random_start=RANDOM_START,
        goal=GOAL,
    )
    obs, _ = env.reset()

    model = MLP_model(SIZE, GOAL, HIDDEN_SIZE, N_ACTIONS)
    policy = Policy(model)

    runner = EnvRunner(
        env,
        policy,
        nsteps=NSTEPS,
        transforms=[ComputeValueTargets(policy), MergeTimeBatch()],
    )

    opt = torch.optim.Adam(
        policy.model.parameters(), lr=LR, betas=(BETA1, BETA2), eps=EPS
    )
    a2c = A2C(policy, opt, VALUE_LOSS_COEF, ENTROPY_COEF)

    s_history = []
    a_history = []
    r_history = []
    t_history = []

    for epoch in trange(NUM_EPOCH):
        trajectory = runner.get_next()

        s_history.append(trajectory["observations"])
        a_history.append(trajectory["actions"])
        r_history.append(trajectory["rewards"])
        t_history.append(trajectory["timesteps"])

        data = a2c.step(trajectory)

        for stat_name in data:
            env.writer.add_scalar(
                f"train/{stat_name}", data[stat_name], runner.step_var
            )

    visualize_policy(policy)

    s_history = np.hstack(s_history)
    a_history = np.hstack(a_history)
    r_history = np.hstack(r_history)
    t_history = np.hstack(t_history)

    env.close()

    if not os.path.exists("src/histories"):
        os.mkdir("src/histories")

    history = np.vstack((s_history, a_history, r_history, t_history))
    np.savetxt(f"src/histories/{goal_name}.txt", history, fmt="%s", delimiter=",")
