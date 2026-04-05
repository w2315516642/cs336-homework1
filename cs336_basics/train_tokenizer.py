from typing import List
from .tokenizer import BPETokenizerTrainer as Trainer
from .tokenizer import BPETokenizer

def check_special_tokens(special_tokens: List[str] = None) -> List[str]:
    if not special_tokens:
        special_tokens = ["<|endoftext|>"]
    else:
        if "<|endoftext|>" not in special_tokens:
            special_tokens.append("<|endoftext|>")
    special_tokens.sort(key=len, reverse=True)
    return special_tokens

def train_bpe(
    file_path: str, 
    vocab_size: int, 
    special_tokens: List[str] = None,
    save_path: str = "params.bin"
) -> None:
    special_tokens = check_special_tokens(special_tokens)

    trainer = Trainer(
        file_path=file_path, 
        vocab_size=vocab_size, 
        special_tokens=special_tokens)

    trainer.train_bpe()
    trainer.save(save_path)

def valid_bpe(
    params_path: str,
    file_path: str,
    special_tokens: List[str] = None
) -> None:
    special_tokens = check_special_tokens(special_tokens)

    bpe = BPETokenizer.from_files(params_path, special_tokens)

    with open(file_path, 'r', encoding='utf-8') as f:
        token_ids = []
        for token_id in bpe.encode_iterable(f):
            token_ids.append(token_id)
        
        f.close()
    print(token_ids[:100])
    


from pathlib import Path
def train_TinyStoriesV2_corpus():
    file_path = Path(__file__).parent.parent / "data" / "TinyStoriesV2-GPT4-train.txt"
    vocab_size = 10000
    save_path = "TinyStoriesV2_params.bin"

    train_bpe(file_path, vocab_size, [], save_path)
    valid_TinyStoriesV2_corpus()


def valid_TinyStoriesV2_corpus():
    params_path = "TinyStoriesV2_params.bin"
    file_path = Path(__file__).parent.parent / "data" / "TinyStoriesV2-GPT4-valid.txt"
    valid_bpe(params_path, file_path)


if __name__ == "__main__":
    train_TinyStoriesV2_corpus()