import numpy as np
import torch


class ComputeValueTargets:
    def __init__(self, policy, gamma=0.99):
        self.policy = policy
        self.gamma = gamma

    def __call__(self, trajectory):
        value_targets = [torch.from_numpy(trajectory['rewards'][-1]).to(torch.float32)]

        last_value = self.policy.act(trajectory['state']['latest_observation'])['values'].flatten()
        last_reset = torch.from_numpy(trajectory['resets'][-1]).to(torch.float32)

        value_targets[-1] += (1 - last_reset) * self.gamma * last_value
        
        for i in range(len(trajectory['resets']) - 1, 0, -1):
            rewards = torch.from_numpy(trajectory['rewards'][i - 1]).to(torch.float32)
            resets = torch.from_numpy(trajectory['resets'][i - 1]).to(torch.float32)
            is_not_done = 1 - resets
            target_ = rewards + is_not_done * self.gamma * value_targets[-1]
            value_targets.append(target_)
        
        value_targets = value_targets[::-1]
        trajectory['value_targets'] = value_targets


class MergeTimeBatch:
    def __call__(self, trajectory):
        trajectory['values'] = list(map(lambda x: x.flatten(), trajectory['values']))
        
        for key in ['values', 'value_targets', 'log_probs']:
            trajectory[key] = torch.vstack(trajectory[key]).reshape(-1, 1)
        
        for key in ['observations', 'actions', 'rewards', 'timesteps']:
            trajectory[key] = np.vstack(trajectory[key]).T.flatten()