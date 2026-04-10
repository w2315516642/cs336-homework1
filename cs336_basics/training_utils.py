import torch
from torch import nn
from torch.nn import functional as F

def softmax(x: torch.Tensor, dim: int=-1) -> torch.Tensor:
    max_val = torch.max(x, dim=dim, keepdim=True).values
    x = x - max_val
    x = torch.exp(x)
    sum_val = torch.sum(x, dim=dim, keepdim=True)
    return x / sum_val

def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    m = inputs.max()
    x = inputs.gather(-1, targets.unsqueeze(-1)) - m
    log_sum = (inputs - m).exp().sum(dim=-1).log()
    loss = (log_sum - x).mean()
    return loss


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

    targets = torch.tensor([[1, 0, 2, 2], [4, 1, 4, 0]])
    print(inputs.shape, targets.shape)
    actual = cross_entropy(inputs.view(-1, inputs.size(-1)), targets.view(-1))
    expected = F.cross_entropy(inputs.view(-1, inputs.size(-1)), targets.view(-1))
    errors = actual - expected

    print(f"actual val: {actual}")
    print(f"expected val: {expected}")
    print(f"errors norm: {errors.norm()}")