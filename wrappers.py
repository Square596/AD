from toymeta.dark_room import DarkRoom
from env_batch import ParallelEnvBatch

# the following code is adapted from https://github.com/yandexdataschool/Practical_RL


class _thunk:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self):
        return make_parallel_env(**self.kwargs)


def make_parallel_env(nenvs=None, seed=None, **kwargs):
    if nenvs is not None:
        if isinstance(seed, int):
            seed = [seed] * nenvs
        if len(seed) != nenvs:
            raise ValueError("len(seed) != nenvs")
        
        thunks = [_thunk(**kwargs) for _ in range(nenvs)]
        env = ParallelEnvBatch(make_env=thunks, seeds=seed)
        return env
    
    env = DarkRoom(**kwargs)

    return env