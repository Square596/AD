import torch


class ComputeValueTargets:
    def __init__(self, policy, gamma=0.99, device=torch.device('cpu')):
        self.policy = policy
        self.gamma = gamma
        self.device = device

    def __call__(self, trajectory):
        value_targets = [torch.tensor(trajectory['rewards'][-1], dtype=torch.float32).to(self.device)]

        last_value = self.policy.act(trajectory['state']['latest_observation'])['values'].flatten()
        last_reset = torch.tensor(trajectory['resets'][-1], dtype=torch.float32).to(self.device)

        value_targets[-1] += (1 - last_reset) * self.gamma * last_value
        
        for i in range(len(trajectory['resets']) - 1, 0, -1):
            rewards = torch.tensor(trajectory['rewards'][i - 1], dtype=torch.float32).to(self.device)
            resets = torch.tensor(trajectory['resets'][i - 1], dtype=torch.float32).to(self.device)
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