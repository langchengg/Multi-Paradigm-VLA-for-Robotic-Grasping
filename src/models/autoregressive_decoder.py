from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.action_tokenizer import ActionTokenizer


class AutoregressiveActionDecoder(nn.Module):
    """Sequential token decoder over discretized action chunks."""

    def __init__(
        self,
        condition_dim: int,
        action_dim: int = 7,
        horizon: int = 16,
        num_bins: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.horizon = int(horizon)
        self.sequence_length = self.action_dim * self.horizon
        self.tokenizer = ActionTokenizer(action_dim=action_dim, num_bins=num_bins)
        self.num_bins = int(num_bins)

        self.condition_proj = nn.Linear(condition_dim, hidden_dim * num_layers)
        self.token_embed = nn.Embedding(num_bins + 1, hidden_dim)
        self.position_embed = nn.Embedding(self.sequence_length, hidden_dim)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_dim, num_bins)
        self.num_layers = int(num_layers)
        self.hidden_dim = int(hidden_dim)

    def forward(self, condition: torch.Tensor, actions: torch.Tensor):
        tokens = self.tokenizer.flatten(self.tokenizer.encode(actions))
        batch_size, seq_len = tokens.shape
        bos = torch.full(
            (batch_size, 1),
            self.tokenizer.bos_token_id,
            dtype=torch.long,
            device=tokens.device,
        )
        inputs = torch.cat([bos, tokens[:, :-1]], dim=1)
        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(batch_size, -1)
        hidden0 = self.condition_proj(condition).view(batch_size, self.num_layers, self.hidden_dim)
        hidden0 = hidden0.transpose(0, 1).contiguous()
        x = self.token_embed(inputs) + self.position_embed(positions)
        output, _ = self.gru(x, hidden0)
        logits = self.output(output)
        loss = F.cross_entropy(logits.reshape(-1, self.num_bins), tokens.reshape(-1))
        return loss, {"autoregressive_ce": float(loss.detach().cpu())}

    @torch.no_grad()
    def sample(self, condition: torch.Tensor, num_steps: int | None = None):
        batch_size = condition.shape[0]
        device = condition.device
        hidden = self.condition_proj(condition).view(batch_size, self.num_layers, self.hidden_dim)
        hidden = hidden.transpose(0, 1).contiguous()
        token = torch.full(
            (batch_size, 1),
            self.tokenizer.bos_token_id,
            dtype=torch.long,
            device=device,
        )
        generated = []
        for pos in range(self.sequence_length):
            position = torch.full((batch_size, 1), pos, dtype=torch.long, device=device)
            x = self.token_embed(token) + self.position_embed(position)
            out, hidden = self.gru(x, hidden)
            logits = self.output(out[:, -1])
            token = logits.argmax(dim=-1, keepdim=True)
            generated.append(token)
        sequence = torch.cat(generated, dim=1)
        token_grid = self.tokenizer.unflatten(sequence, self.horizon)
        return self.tokenizer.decode(token_grid)

