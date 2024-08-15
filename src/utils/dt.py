import random
from glob import glob
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset

# the following code is adapted from https://github.com/corl-team/CORL/blob/main/algorithms/offline/dt.py


def pad_along_axis(
    arr: np.ndarray, pad_to: int, axis: int = 0, fill_value: float = 0.0
) -> np.ndarray:
    pad_size = pad_to - arr.shape[axis]
    if pad_size <= 0:
        return arr

    npad = [(0, 0)] * arr.ndim
    npad[axis] = (0, pad_size)
    return np.pad(arr, pad_width=npad, mode="constant", constant_values=fill_value)


def discounted_cumsum(x: np.ndarray, gamma: float) -> np.ndarray:
    cumsum = np.zeros_like(x)
    cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        cumsum[t] = x[t] + gamma * cumsum[t + 1]
    return cumsum


class SequenceDataset(IterableDataset):
    def __init__(
        self,
        history_path: str,
        history_len: int | None = None,
        seq_len: int = 10,
        time_rel: bool = True,
    ):

        if history_len is None:
            loadtxt_fn = lambda filename: np.loadtxt(filename, delimiter=",").astype(
                int
            )
        else:
            loadtxt_fn = lambda filename: np.loadtxt(
                filename, delimiter=",", usecols=np.arange(history_len)
            ).astype(int)

        self.dataset = torch.from_numpy(
            np.stack(
                list(
                    map(
                        loadtxt_fn,
                        glob(f"{history_path}/*.txt"),
                    )
                )
            )
        )

        self.seq_len = seq_len
        self.time_rel = time_rel  # type of time encoding: relative (time step in an env) or absolute (position in a history)

    def __prepare_sample(self, traj_idx, start_idx):
        states = self.dataset[traj_idx, 0, start_idx : start_idx + self.seq_len].to(
            torch.int
        )
        actions = self.dataset[traj_idx, 1, start_idx : start_idx + self.seq_len].to(
            torch.int
        )
        returns = self.dataset[traj_idx, 2, start_idx : start_idx + self.seq_len].to(
            torch.int
        )
        if self.time_rel:
            time_steps = self.dataset[
                traj_idx, 3, start_idx : start_idx + self.seq_len
            ].to(torch.int)
        else:
            time_steps = torch.arange(
                start_idx, start_idx + self.seq_len, dtype=torch.int
            )

        # pad up to seq_len if needed, padding is masked during training
        mask = np.hstack(
            [np.ones(states.shape[0]), np.zeros(self.seq_len - states.shape[0])]
        )
        if states.shape[0] < self.seq_len:
            states = pad_along_axis(states, pad_to=self.seq_len)
            actions = pad_along_axis(actions, pad_to=self.seq_len)
            returns = pad_along_axis(returns, pad_to=self.seq_len)

        return states, actions, returns, time_steps, mask

    def __iter__(self):
        while True:
            traj_idx = random.choice(range(self.dataset.shape[0]))
            start_idx = random.randint(0, self.dataset.shape[-1] - self.seq_len - 2)
            yield self.__prepare_sample(traj_idx, start_idx)


# Decision Transformer implementation
class TransformerBlock(nn.Module):
    def __init__(
        self,
        seq_len: int,
        embedding_dim: int,
        num_heads: int,
        attention_dropout: float,
        residual_dropout: float,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.drop = nn.Dropout(residual_dropout)

        self.attention = nn.MultiheadAttention(
            embedding_dim, num_heads, attention_dropout, batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 32 * embedding_dim),
            nn.GELU(),
            nn.Linear(32 * embedding_dim, embedding_dim),
            nn.Dropout(residual_dropout),
        )
        # True value indicates that the corresponding position is not allowed to attend
        self.register_buffer(
            "causal_mask", ~torch.tril(torch.ones(seq_len, seq_len)).to(bool)
        )
        self.seq_len = seq_len

    # [batch_size, seq_len, emb_dim] -> [batch_size, seq_len, emb_dim]
    def forward(
        self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        causal_mask = self.causal_mask[: x.shape[1], : x.shape[1]]

        norm_x = self.norm1(x)
        attention_out = self.attention(
            query=norm_x,
            key=norm_x,
            value=norm_x,
            attn_mask=causal_mask,
            key_padding_mask=padding_mask,
            need_weights=False,
        )[0]
        # by default pytorch attention does not use dropout
        # after final attention weights projection, while minGPT does:
        # https://github.com/karpathy/minGPT/blob/7218bcfa527c65f164de791099de715b81a95106/mingpt/model.py#L70 # noqa
        x = x + self.drop(attention_out)
        x = x + self.mlp(self.norm2(x))
        return x


class DecisionTransformer(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        reward_nunique: int,
        seq_len: int = 10,
        episode_len: int = 20,
        time_rel: bool = True,
        embedding_dim: int = 64,
        num_layers: int = 4,
        num_heads: int = 4,
        attention_dropout: float = 0.5,
        residual_dropout: float = 0.0,
        embedding_dropout: float = 0.0,
    ):
        super().__init__()
        self.emb_drop = nn.Dropout(embedding_dropout)
        self.emb_norm = nn.LayerNorm(embedding_dim)

        self.out_norm = nn.LayerNorm(embedding_dim)
        # additional seq_len embeddings for padding timesteps
        if time_rel:
            self.timestep_emb = nn.Embedding(episode_len, embedding_dim)
        else:
            self.timestep_emb = nn.Embedding(episode_len + seq_len, embedding_dim)
        self.state_emb = nn.Embedding(state_dim, embedding_dim)
        self.action_emb = nn.Embedding(action_dim, embedding_dim)
        self.return_emb = nn.Embedding(reward_nunique, embedding_dim)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    seq_len=3 * seq_len,
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    attention_dropout=attention_dropout,
                    residual_dropout=residual_dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.action_head = nn.Linear(embedding_dim, action_dim)
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.episode_len = episode_len

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
        self,
        states: torch.Tensor,  # [batch_size, seq_len, state_dim]
        actions: torch.Tensor,  # [batch_size, seq_len, action_dim]
        returns: torch.Tensor,  # [batch_size, seq_len]
        time_steps: torch.Tensor,  # [batch_size, seq_len]
        padding_mask: Optional[torch.Tensor] = None,  # [batch_size, seq_len]
    ) -> torch.FloatTensor:
        batch_size, seq_len = states.shape[0], states.shape[1]
        # [batch_size, seq_len, emb_dim]
        time_emb = self.timestep_emb(time_steps)
        state_emb = self.state_emb(states) + time_emb
        act_emb = self.action_emb(actions) + time_emb
        returns_emb = self.return_emb(returns) + time_emb

        # [batch_size, seq_len * 3, emb_dim], (r_0, s_0, a_0, r_1, s_1, a_1, ...)
        sequence = (
            torch.stack([returns_emb, state_emb, act_emb], dim=1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 3 * seq_len, self.embedding_dim)
        )
        if padding_mask is not None:
            # [batch_size, seq_len * 3], stack mask identically to fit the sequence
            padding_mask = (
                torch.stack([padding_mask, padding_mask, padding_mask], dim=1)
                .permute(0, 2, 1)
                .reshape(batch_size, 3 * seq_len)
            )
        # LayerNorm and Dropout (!!!) as in original implementation,
        # while minGPT & huggingface uses only embedding dropout
        out = self.emb_norm(sequence)
        out = self.emb_drop(out)

        for block in self.blocks:
            out = block(out, padding_mask=padding_mask)

        out = self.out_norm(out)
        # [batch_size, seq_len, action_dim]
        # predict actions only from state embeddings
        out = self.action_head(out[:, 1::3])
        return out


class DT_Trainer:
    def __init__(self, model, optimizer, scheduler, loss_fn, max_grad_norm):
        self.model = model
        self.opt = optimizer
        self.max_grad_norm = max_grad_norm
        self.scheduler = scheduler
        self.loss_fn = loss_fn

    def step(self, batch):
        self.model.train()
        states, actions, returns, time_steps, mask = batch
        padding_mask = ~mask.to(torch.bool)
        pred = self.model(
            states=states,
            actions=actions,
            returns=returns,
            time_steps=time_steps,
            padding_mask=padding_mask,
        )

        loss = self.loss_fn(
            pred.view(-1, self.model.action_dim), actions.view(-1).to(torch.long)
        )
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.max_grad_norm
        ).item()

        self.opt.step()
        self.opt.zero_grad()

        self.scheduler.step()

        return {
            "grad_norm": grad_norm,
            "loss": loss.detach().item(),
        }
