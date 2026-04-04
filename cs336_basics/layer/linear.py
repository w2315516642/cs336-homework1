import torch
from torch import nn

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None) -> None:
        super().__init__()

        self.device: torch.device = device if device else "cpu"
        self.dtype: torch.dtype = dtype if dtype else torch.float32

        # 参数初始化
        self.weight = nn.Parameter(torch.rand(
                size=(in_features, out_features), 
                device=self.device, 
                dtype=self.dtype
            ))
        # 参数截断
        sigma = torch.sqrt(torch.tensor(2.0 / (in_features + out_features)))
        self.weight = nn.init.trunc_normal_(
            self.weight, std=sigma,a=-3 * sigma, b=3 * sigma
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x @ self.weight
        return out
        

if __name__ == "__main__":
    in_features = 64
    out_features = 256
    model = Linear(in_features, out_features)

    model_ = nn.Linear(in_features, out_features, bias=False)

    x = torch.rand((1, in_features))
    y = model(x)
    y_ = model_(x)

    print(f"std1: {torch.std(y)}, std2: {torch.std(y_)}")