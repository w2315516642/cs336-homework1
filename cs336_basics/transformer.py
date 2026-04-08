import torch
from torch import nn
from typing import List

from .layers.embedding import Embedding
from .layers.transformer_block import TransformerBlock
from .layers.scale_dot_product_attention import softmax
from .layers.linear import Linear
from .layers.rmsnorm import RMSNorm

class Transformer(nn.Module):
    def __init__(
        self, 
        vocab_size: int, 
        context_length: int, 
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: int,
        device: torch.device,
        dtype: torch.dtype
    ) -> None:
        super().__init__()

        kwargs = {"device": device, "dtype": dtype}

        self.embedding = Embedding(vocab_size, d_model, **kwargs)
        
        self.transformers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, context_length, theta, **kwargs)
            for _ in range(num_layers)
        ])

        self.norm = RMSNorm(d_model, **kwargs)
        self.linear = Linear(d_model, vocab_size, **kwargs)
    
    def forward(self, token_ids: torch.Tensor | List[int]) -> torch.Tensor:
        if not isinstance(token_ids, torch.Tensor):
            token_ids = torch.tensor(token_ids) 
        # embedding
        x = self.embedding(token_ids)

        # forward
        for layer in self.transformers:
            x = layer(x)
        x = self.norm(x)
        x = self.linear(x)
        return softmax(x)