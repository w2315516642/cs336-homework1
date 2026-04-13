import torch
from torch import nn
from typing import List

from .layers.embedding import Embedding
from .layers.transformer_block import TransformerBlock
from .layers.position_encoding import RotaryPositionalEmbedding as RoPE
from .layers.linear import Linear
from .layers.rmsnorm import RMSNorm

from .configs import Config

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
        device: torch.device=None,
        dtype: torch.dtype=None
    ) -> None:
        super().__init__()

        kwargs = {"device": device, "dtype": dtype}

        self.embedding = Embedding(vocab_size, d_model, **kwargs)
        self.rope = RoPE(d_model // num_heads, context_length, theta, **kwargs)
        
        self.transformers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, **kwargs)
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
            x = layer(x, self.rope)
        x = self.norm(x)
        x = self.linear(x)
        "注意这里不需要 softmax"
        return x
    
    @classmethod
    def from_config(cls, config: Config):
        model_config = config.model
        return cls(
            vocab_size=model_config.vocab_size,
            context_length=model_config.context_length,
            num_layers=model_config.num_layers,
            d_model=model_config.d_model,
            num_heads=model_config.num_heads,
            d_ff=model_config.d_ff,
            theta=model_config.theta,
            device=config.device,
            dtype=config.dtype
        )