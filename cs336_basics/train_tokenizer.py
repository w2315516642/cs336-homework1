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
    num_processor: int = 6,
    save_path: str = "params.bin"
) -> None:
    special_tokens = check_special_tokens(special_tokens)

    trainer = Trainer(
        file_path=file_path, 
        vocab_size=vocab_size, 
        special_tokens=special_tokens,
        num_processor=num_processor)

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
    parser = get_parser()
    args = parser.parse_args()

    file_path = Path(__file__).parent.parent / "data" / args.train_file
    vocab_size = args.vocab_size
    save_path = args.save_name
    print(file_path, vocab_size, save_path)
    train_bpe(file_path, vocab_size, ['<|endoftext|>'], args.num_processor, save_path)
    valid_TinyStoriesV2_corpus()


def valid_TinyStoriesV2_corpus():
    parser = get_parser()
    args = parser.parse_args()

    params_path = args.save_name
    file_path = Path(__file__).parent.parent / "data" / args.valid_file
    valid_bpe(params_path, file_path)


import argparse
def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-nc', '--num_processor', type=int, help='使用cpu的数量', default=6)
    parser.add_argument('-vs', '--vocab_size', type=int, help='词汇表大小', default=10000)
    parser.add_argument('-tf', '--train_file', type=str, help='训练文件名字', default="TinyStoriesV2-GPT4-train.txt")
    parser.add_argument('-vf', '--valid_file', type=str, help='验证文件名字', default="TinyStoriesV2-GPT4-valid.txt")
    parser.add_argument('-sn', '--save_name', type=str, help='保存文件名字', default="TinyStoriesV2_params.bin")
    return parser


if __name__ == "__main__":
    train_TinyStoriesV2_corpus()