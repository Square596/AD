from toymeta.dark_room import DarkRoom

from src.utils.env_batch import ParallelEnvBatch

# the following code is adapted from https://github.com/yandexdataschool/Practical_RL


class DarkRoom_with_time(DarkRoom):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.time = None
    
    def reset(self, *args, **kwargs):
        s, info = super().reset(*args, **kwargs)
        self.time = 1
        info['tmp'] = self.time
        return s, info
    
    def step(self, *args, **kwargs):
        s, r, terminated, truncated, info = super().step(*args, **kwargs)
        self.time += 1
        info['tmp'] = self.time
        return s, r, terminated, truncated, info
    

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
    
    env = DarkRoom_with_time(**kwargs)

    return env