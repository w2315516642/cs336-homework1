import math
from pathlib import Path
from typing import Iterable, List, Tuple
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

def softmax(x: torch.Tensor, dim: int=-1) -> torch.Tensor:
    in_dtype = x.dtype
    x.to(torch.float32)
    max_val = torch.max(x, dim=dim, keepdim=True).values
    x = x - max_val
    x = torch.exp(x)
    sum_val = torch.sum(x, dim=dim, keepdim=True)
    o = x / sum_val
    return o.to(in_dtype)

def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    m, _ = inputs.max(dim=-1, keepdim=True)
    # print(f"max val in inputs: {m}")
    x = inputs.gather(-1, targets.unsqueeze(-1)) - m
    # print(f"target val: {x}")
    inputs = inputs - m
    # print(f"inputs - m = {inputs}")
    log_sum = inputs.exp().sum(dim=-1, keepdim=True).log()
    # print(f"log_sum: {log_sum}")
    loss = (log_sum - x).mean()
    # print(f"final loss: {loss}")
    return loss


def perplexity(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    "input: [batch, seq_len, vocab_size]"
    return cross_entropy(inputs, targets).exp()


def cosine_learning_rate_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    # warm-up
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate
    # cosine annealing
    if it >= warmup_iters and it <= cosine_cycle_iters:
        at = math.cos(
            (it - warmup_iters) 
            / (cosine_cycle_iters - warmup_iters) 
            * math.pi
        )
        at = 0.5 * (1 + at) * (max_learning_rate - min_learning_rate)
        return min_learning_rate + at
    # post annealing
    return min_learning_rate


def gradient_clipping(
    params: Iterable[nn.Parameter], 
    max_l2_norm: float,
    eps: float=1e-6
) -> None:
    grads: List[torch.Tensor] = [p.grad for p in params if p.grad is not None]
    if not grads:
        return
    
    l2_norm = 0.0
    for g in grads:
        l2_norm += torch.sum(g ** 2)
    l2_norm = torch.sqrt(l2_norm)

    if l2_norm > max_l2_norm:
        clip_coef = min(1.0, max_l2_norm / (l2_norm + eps))
        for g in grads:
            g *= clip_coef


def get_batch(
    dataset: np.ndarray, 
    batch_size: int, 
    context_length: int, 
    device: str=None
) -> Tuple[torch.Tensor, torch.Tensor]:
    max_start = len(dataset) - context_length - 1
    assert max_start > 0, f"""The length of dataset {len(dataset)} 
    smaller than context length {context_length}."""

    starts = np.random.randint(0, max_start + 1, size=batch_size)
    x_batched = []
    y_batched = []
    for s in starts:
        seq = dataset[s : s + context_length + 1]
        x_batched.append(seq[:-1])
        y_batched.append(seq[1:])

    x = torch.tensor(np.array(x_batched), dtype=torch.long, device=device)
    y = torch.tensor(np.array(y_batched), dtype=torch.long, device=device)
    return x, y


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    save_path: str | Path,
) -> None:
    checkpoint = {
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "iteration": iteration
    }
    torch.save(checkpoint, save_path)


def load_checkpoint(
    file_path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optim_state"])
    return checkpoint["iteration"]


if __name__ == "__main__":
    inputs = torch.tensor(
        [
            [
                [0.1088, 0.1060, 0.6683, 0.5131, 0.0645],
                [0.4538, 0.6852, 0.2520, 0.3792, 0.2675],
                [0.4578, 0.3357, 0.6384, 0.0481, 0.5612],
                [0.9639, 0.8864, 0.1585, 0.3038, 0.0350],
            ],
            [
                [0.3356, 0.9013, 0.7052, 0.8294, 0.8334],
                [0.6333, 0.4434, 0.1428, 0.5739, 0.3810],
                [0.9476, 0.5917, 0.7037, 0.2987, 0.6208],
                [0.8541, 0.1803, 0.2054, 0.4775, 0.8199],
            ],
        ]
    )
    inputs = 1000 * inputs
    targets = torch.tensor([[1, 0, 2, 2], [4, 1, 4, 0]])
    print(inputs.shape, targets.shape)
    actual = cross_entropy(inputs.view(-1, inputs.size(-1)), targets.view(-1))
    expected = F.cross_entropy(inputs.view(-1, inputs.size(-1)), targets.view(-1))
    errors = actual - expected

    print(f"actual val: {actual}")
    print(f"expected val: {expected}")
    print(f"errors norm: {errors.norm()}")