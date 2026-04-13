import os
from pathlib import Path
from typing import List, Dict, Any
import chardet
import re
import yaml


def read_yaml(file_path: str) -> Dict[str, Any]:
    if not Path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} is not exists.")
    # 读取 yaml 为 str
    content = load_yaml_stably(file_path)
    if not content:
        raise IOError(f"Failed to read configuration file {file_path}")
    
    # 替换环境变量
    pattern = re.compile(r"\$\{(\w+)\}")
    def replacer(match: re.Match):
        env_var = match.group(1)
        return os.getenv(env_var, match.group(0))
    content = pattern.sub(replacer, content)

    # 读取文件
    try:
        return yaml.safe_load(content)
    except yaml.YAMLError as e:
        print(f"Error in reading yaml file: {e}")
        raise e


def load_yaml_stably(file_path: str) -> str | None:

    encodings = ["utf-8", "utf-8-sig", "gbk", "gb2312", "ascii", "cp936"]
    for encoding in encodings:
        try:
            with open(file_path, "r", encoding=encoding) as f:
                print(f"Successfully read file {file_path} with encoding {encoding}")
                return f.read()
        except UnicodeDecodeError:
            continue
    # 尝试二进制
    try:
        with open(file_path, "rb") as f:
            raw_data = f.read()
        detect = chardet.detect(raw_data)
        if detect["encoding"]:
            return raw_data.decode(detect["encoding"])
    except Exception as e:
        print(f"Can not read file {file_path} with right encoding: {e}")
    return None

def find_chunk_boundaries(
    file_path: str,
    desired_num_chunks: int,
    split_spectial_token: bytes,
) -> List[int]:
    """
    将文本文件切分成可以被单独统计的块，返回分块边界
    """

    assert isinstance(split_spectial_token, bytes), "分割字符必须为字节形式"

    with open(file_path, "rb") as file:
        # 统计文件大小（字节数）
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        # 初始化边界位置
        chunk_size = file_size // desired_num_chunks
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks)]
        chunk_boundaries[-1] = file_size

        # 从每个初始位置开始往后搜索特殊分割字符，作为实际分割位置
        # 每次读取的字节数
        mini_chunk_size = 4096

        for bi in range(1, len(chunk_boundaries) - 1):
            # 初始搜索位置
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)
            
            while True:
                # 读取部分数据
                mini_chunk = file.read(mini_chunk_size)

                # 若已经到文件末尾，则分割完毕
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break

                # 搜索特殊分割字符
                found_at = mini_chunk.find(split_spectial_token)
                # 若找到则退出循环
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break

                initial_position += mini_chunk_size
        # 多个初始边界点搜到同一个位置，需要确保唯一性
    return sorted(set(chunk_boundaries))


def print_max_n(dictionary: Dict[str, int], n: int, idx_key: int = 1) -> None:
    sorted_dict = sorted(dictionary.items(), key=lambda x: x[idx_key], reverse=True)
    for i in range(n):
        print(sorted_dict[i])


def make_divisible(x: int, divisor: int=64, min_value: int=None) -> int:
    if min_value is None:
        min_value = divisor
    # 四舍五入
    new_x = max(min_value, int(x + divisor / 2) // divisor * divisor)
    if new_x < 0.9 * x:
        new_x += divisor
    return new_x