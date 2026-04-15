import torch
from pathlib import Path
from typing import List

from cs336_basics.transformer import Transformer
from cs336_basics.tokenizer import BPETokenizer
from cs336_basics.configs import Config
from cs336_basics.training_utils import load_checkpoint


def generate(
    model: Transformer,
    tokenizer: BPETokenizer,
    prompts: str | List[str],
    temperature: float = 1,
    top_p: float | None = None,
) -> str | List[str]:
    if isinstance(prompts, str):
        prompts = [prompts]
    
    tokens_list = []
    for prompt in prompts:
        tokens_list.append(tokenizer.encode(prompt))

    out_tokens = model.generate(
        tokens_list,
        temperature=temperature,
        top_p=top_p,
        max_len=10
     )

    outputs = []
    for tokens in out_tokens:
        outputs.append(tokenizer.decode(tokens))
    return outputs


def load_model(file_path: str | Path, config: Config) -> Transformer:
    if isinstance(file_path, str):
        file_path = Path(file_path)
    ckpt_files = list(file_path.glob("checkpoint_*.pt"))
    ckpt_files.sort(key= lambda x: int(x.stem.split('_')[1]), reverse=True)
    # 取出训练轮数最多的那个
    ckpt_file = ckpt_files[0]

    model = Transformer.from_config(config)
    load_checkpoint(ckpt_file, model, device=config.device)
    return model


def main_pipeline():
    model_path = Path('output/2026_0414_2014_owt/')
    tokenizer_path = "cs336_basics/owt_params.bin"

    config = Config.from_yaml(Path("config.yaml"))
    config.device = 'cpu'
    model = load_model(model_path, config)
    tokenizer = BPETokenizer.from_files(tokenizer_path, ['<|endoftext|>'])

    prompts = [
        "introduce yourself"
    ]

    outputs = generate(model, tokenizer, prompts)
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main_pipeline()