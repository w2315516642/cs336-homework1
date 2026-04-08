import torch
from torch import nn

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None) -> None:
        super().__init__()

        kwargs = {"device": device, "dtype": dtype}
        
        # multipler = 64
        # d_ff = math.ceil(d_ff / multipler) * multipler
        
        self.weight1 = nn.Parameter(torch.empty((d_ff, d_model), **kwargs))
        self.weight2 = nn.Parameter(torch.empty((d_model, d_ff), **kwargs))
        self.weight3 = nn.Parameter(torch.empty((d_ff, d_model), **kwargs))

        sigma = torch.sqrt(torch.tensor(2 / (d_model + d_ff)))
        self.weight1 = nn.init.trunc_normal_(self.weight1, std=sigma, a=-3 * sigma, b=3 * sigma)
        self.weight2 = nn.init.trunc_normal_(self.weight2, std=sigma, a=-3 * sigma, b=3 * sigma)
        self.weight3 = nn.init.trunc_normal_(self.weight3, std=sigma, a=-3 * sigma, b=3 * sigma)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        y = self.Swish(x @ self.weight1.T)
        v = x @ self.weight3.T
        out = (y * v) @ self.weight2.T
        return out.to(in_dtype)
    
    @staticmethod
    def Swish(x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


if __name__ == "__main__":
    d_model = 64
    d_ff = 128
    model = SwiGLU(d_model, d_ff)

    x = torch.randn((6, 12, d_model), dtype=torch.float16)

    y = model(x)
    
    