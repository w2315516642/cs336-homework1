
if __name__ == "__main__":
    # 参数设置
    vocab_size = 32000
    context_length = 256
    num_layers = 4
    
    d_model = 512
    num_heads = 16
    d_ff = 1344
    is_swiglu = True

    # 参数量计算（没有 bias）
    # embedding + RoPE层参数量
    embedding = vocab_size * d_model
    rope = context_length * (d_model / num_heads) * 2 / 2
    
    # 一个 transformer block 内的参数量
    rms = d_model * 2  # 两个 RMS 块
    attn_qkvo = d_model * d_model * 4
    if is_swiglu:
        fnn = d_model * d_ff * 2 + d_ff * d_model
    else:
        fnn = d_model * d_ff * 2
        
    # transformer block 的总参数量
    params_per_tb = rms + attn_qkvo + fnn
    tb = params_per_tb * num_layers

    # 输出为一个 RMS 层 + 一个 Linear 层
    out_layer = d_model + d_model * vocab_size

    total_params = int(embedding + rope + tb + out_layer)

    print(f"total params is {total_params}, params size in bfloat16 is {total_params * 2 / 1024 ** 2} MB")
