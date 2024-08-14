from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from torch.distributions import Categorical


class MLP_model(nn.Module):
    def __init__(self, size, goal, emb_dim, n_actions):
        super().__init__()
        self.size = size
        self.goal = goal

        self.emb = nn.Embedding(size * size, emb_dim)
        self.norm1 = nn.BatchNorm1d(emb_dim)
        self.act1 = nn.Tanh()

        self.actor_critic = nn.Linear(emb_dim, n_actions + 1)

    def forward(self, inp):
        inp = torch.from_numpy(inp)
        out = self.act1(self.norm1(self.emb(inp)))
        out = self.actor_critic(out)

        values = out[:, 0]
        logits = out[:, 1:]

        return values, logits


class Policy:
    def __init__(self, model):
        self.model = model
        self.size = model.size
        self.goal = model.goal

    def act(self, inputs):
        values, logits = self.model(inputs)

        distributions = Categorical(logits=logits)
        actions = distributions.sample()
        log_probs = distributions.log_prob(actions)
        actions = np.array(actions)

        return {
            "actions": actions,
            "logits": logits,
            "log_probs": log_probs,
            "values": values,
            "distributions": distributions,
        }


class A2C:
    def __init__(
        self,
        policy,
        optimizer,
        value_loss_coef=0.25,
        entropy_coef=1e-2,
        max_grad_norm=0.5,
    ):
        self.policy = policy
        self.optimizer = optimizer
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

    def policy_loss(self, trajectory):
        """
        policy improvement
        """
        return -torch.mean(
            trajectory["log_probs"]
            * (trajectory["value_targets"].detach() - trajectory["values"].detach())
        )

    def value_loss(self, trajectory):
        """
        policy evaluation
        """
        return torch.mean(
            torch.pow(trajectory["value_targets"].detach() - trajectory["values"], 2)
        )

    def entropy(self, trajectory):
        """
        entropy regularization
        """
        return torch.mean(
            torch.vstack(list(map(lambda x: x.entropy(), trajectory["distributions"])))
        )

    def loss(self, trajectory):
        policy_loss = self.policy_loss(trajectory)
        value_loss = self.value_loss(trajectory)
        entropy = self.entropy(trajectory)

        a2c_loss = (
            policy_loss
            - self.entropy_coef * entropy
            + self.value_loss_coef * value_loss
        )

        return policy_loss, value_loss, entropy, a2c_loss

    def r2(self, trajectory):
        """
        evaluate values
        """
        return r2_score(
            trajectory["value_targets"].detach().numpy(),
            trajectory["values"].detach().numpy(),
        )

    def step(self, trajectory):
        """
        update policy and evaluate
        """
        r2 = self.r2(trajectory)
        policy_loss, value_loss, entropy, a2c_loss = self.loss(trajectory)
        value_targets = trajectory["value_targets"].detach().numpy().mean()
        value_pred = trajectory["values"].detach().numpy().mean()
        advantage = value_targets - value_pred

        a2c_loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.policy.model.parameters(), self.max_grad_norm
        ).item()

        self.optimizer.step()
        self.optimizer.zero_grad()

        return {
            "values_r2": r2,
            "entropy": entropy.detach().item(),
            "value_loss": value_loss.detach().item(),
            "policy_loss": policy_loss.detach().item(),
            "value_targets": value_targets,
            "value_pred": value_pred,
            "grad_norm": grad_norm,
            "advantage": advantage,
            "a2c_loss": a2c_loss.detach().item(),
        }
