import torch
from torch import nn
from .linear import Linear
from .scale_dot_product_attention import ScaleDotProductAttention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.attention = ScaleDotProductAttention()
        self.w_qkv = Linear(d_model, 3 * d_model)
        self.w_o = Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, ..., seq_len, d_model]
        qkv: torch.Tensor = self.w_qkv(x)
        q, k, v = torch.chunk(qkv, chunks=3, dim=-1)
        
        # qkv: [batch, ..., seq_len, d_model] -> [batch, ..., seq_len, num_heads, d_k]
        #                                     -> [batch, ..., num_heads, seq_len, d_k]
        qkv_shape = q.size()[:-1]
        
        q_h = q.view(*qkv_shape, self.num_heads, -1).contiguous().transpose(-2, -1)
        k_h = k.view(*qkv_shape, self.num_heads, -1).contiguous().transpose(-2, -1)
        v_h = v.view(*qkv_shape, self.num_heads, -1).contiguous().transpose(-2, -1)

        d_k = q_h.size()[-1]
        mask = [[True if x <= i else False for x in range(d_k)] for i in range(d_k)]
        mask = torch.tensor(mask)

        # v_h: [batch, ..., num_heads, seq_len, d_k] -> [batch, ..., num_heads, seq_len, d_k]
        new_v_h: torch.Tensor = self.attention(q_h, k_h, v_h, mask)
        # v_h: [batch, ..., num_heads, seq_len, d_k] -> [batch, ..., seq_len, num_heads, d_k]
        #                                            -> [batch, ..., seq_len, num_heads * d_k]
        v_concat = new_v_h.view(*qkv_shape, -1).contiguous()
        out = self.w_o(v_concat)
        return out

if __name__ == "__main__":
    d_model = 64
    num_heads = 8
    model = MultiHeadAttention(d_model, num_heads)

    x = torch.randn((6, 12, d_model))
    out = model(x)
