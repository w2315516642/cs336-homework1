from pathlib import Path
from typing import IO, BinaryIO, Dict, Iterable, Optional, Self, Tuple, List, final
from abc import ABC, abstractmethod
from tqdm import tqdm
import regex as re
import pickle, struct

from multiprocessing import Pool
from collections import Counter
from functools import partial

from .utils import find_chunk_boundaries, print_max_n

# 预分词匹配字符串
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer(ABC):

    @staticmethod
    def pre_tokenize(
        text: str, 
        special_tokens: List[str] | str = []
    ):
        pre_tokens_to_token = {}
        pre_tokens_seq = []

        if special_tokens:
            # 要用re.escape进行特殊字符转义（'|'，'<'等）
            concat_special_token = "|".join([re.escape(token) for token in special_tokens])
            splited_chunk = re.split(concat_special_token, text)
            matchs = re.finditer(concat_special_token, text)
            special_tokens_seq = [match.group() for match in matchs]
        else:
            splited_chunk = [text]
            special_tokens_seq = []

        i = 0
        for seg in splited_chunk:
            pre_token_iter = re.finditer(PAT, seg)
            for pre_token in pre_token_iter:
                # 将字符串转化为字节元组
                pre_token = pre_token.group().encode(encoding="utf-8", errors="ignore")
                pre_token = tuple(bytes([b]) for b in pre_token)
                if pre_token not in pre_tokens_to_token:
                    pre_tokens_to_token[pre_token] = pre_token
                pre_tokens_seq.append(pre_token)
            # 每段分割完后需要把特殊分割符给补上
            if i < len(special_tokens_seq):
                pre_tokens_seq.append(special_tokens_seq[i])
            i += 1

        return pre_tokens_to_token, pre_tokens_seq, special_tokens_seq


    @staticmethod
    def pre_tokenize_itera(text: str) -> Iterable:
        pre_token_iter = re.finditer(PAT, text)
        for pre_token in pre_token_iter:
            # 将字符串形式转变为tuple(bytes)形式，后面合并要用
            pre_token = pre_token.group().encode("utf-8")
            pre_token = tuple(bytes([b]) for b in pre_token)

            yield pre_token

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """将文本编码为整数序列"""
        raise NotImplementedError("encode method must be implemented")

    @abstractmethod
    def decode(self, tokens: List[int]) -> str:
        """将整数序列解码为文本"""
        raise NotImplementedError("decode method must be implemented")


# @dataclass
# class BPETokenizerParams:
#     # 存储BEP分词器的单词表vocab: Dict[int, bytes]
#     # 与整合映射merges: Dict[Tuple[int, int], int]
#     vocab: Dict[int, bytes] = None
#     merges: Dict[Tuple[int, int], int] = None

class BPETokenizerTrainer:
    def __init__(
        self, 
        file_path: str, 
        vocab_size: int, 
        special_tokens: List[str] | str, 
        num_processor: int = 6,
    ) -> None:
        self.file_path = file_path
        self.vocab_size = vocab_size if vocab_size else 512
        self.special_tokens = special_tokens if special_tokens else []
        self.num_processor = num_processor

        # 初始化BPE训练过程中需要的变量
        self.pre_tokens = {}    # 对语料库pre-token后的计数结果
        self.bp_freq = {}       # 字节对频率统计结果

        self.vocab = {}
        self.merges = []

    def train_bpe(self) -> None:

        input_path = self.file_path
        target_vocab_size = self.vocab_size
        special_tokens = self.special_tokens
        # 根据特殊tokens初始化词汇表
        len_st = len(special_tokens)
        vocab = {i: bytes(special_tokens[i], encoding="utf-8") for i in range(len_st)}
        vocab.update({len_st + i: bytes([i]) for i in range(256)})

        # 合并列表初始化
        merges = []

        # 预分词，修改self.pre_tokens
        self.pre_tokens = self.count_pre_tokenize(
            input_path, 
            special_tokens, 
            self.num_processor
        )

        # 根据预分词结果初始化byte-pair出现次数
        self.init_byte_pair_freq()
        # print_max_n(self.bp_freq, 10)

        # 合并，每轮合并一个字节对
        num_merges = max(target_vocab_size - len(vocab), 0)
        for _ in tqdm(range(num_merges), desc="Merging words"):
            self.merge_bpe(vocab, merges)

        self.vocab = vocab
        self.vocab_to_index = {v: k for k, v in self.vocab.items()}
        self.merges = merges 

    # 初始化byte-pair频次
    def init_byte_pair_freq(self) -> None:
        bp_freq = {}
        for idx, pre_token_data in tqdm(
            self.pre_tokens.items(), 
            desc="Initializing bp-freq table"
        ):
            # 取出整词的pre-token和对应的计数
            pre_token, count = pre_token_data
            
            for byte_pair in zip(pre_token[:-1], pre_token[1:]):
                if byte_pair not in bp_freq:
                    bp_freq[byte_pair] = [0, set()]
                bp_freq[byte_pair][0] += count
                bp_freq[byte_pair][1].add(idx)

        self.bp_freq = bp_freq

    def merge_bpe(
        self,
        vocab: Dict[int, Tuple[bytes]],
        merges: List[Tuple[bytes, bytes]],
    ) -> None:
        # 找到出现次数最多的字节对（依次使用元组内三个评判标准）
        pair: Tuple[bytes, bytes] = max(
            self.bp_freq, 
            key=lambda x: (self.bp_freq[x][0], x[0], x[1])
        )

        # 更新词汇表 vocab 和 merges
        new_bytes = pair[0] + pair[1]

        # 词汇表大小作为新的索引值
        vocab_size = len(vocab)
        vocab[vocab_size] = new_bytes

        # merges.append((bytes(pair[0]), bytes(pair[1])))
        merges.append(pair)
        # 更新bp-freq
        self.merge_pre_tokens(pair)

    def merge_pre_tokens(self, pair: Tuple[bytes, bytes]) -> None:
        # 根据新的pair更新bp_freq，需要在pre-tokens里面把出现的pair合并，然后再更新bp_freq
        # 找到本轮pair涉及到的pre-token（这里应该能开多线程，但是bp-freq是共享的，也不好说）
        have_pair_token_indices = self.bp_freq[pair][1]
        new_bytes = pair[0] + pair[1]
        for have_pair_idx in have_pair_token_indices:
            # 获取相关的pre-token和对应计数
            pre_token, count = self.pre_tokens[have_pair_idx]

            # 找到pair在当前pre-token中的位置，并合并pair和前后相邻的bytes
            pos = 0
            while pos < len(pre_token) - 1:
                bp = (pre_token[pos], pre_token[pos + 1])
                # 找到要替换的位置
                if bp == pair:
                    # 若前面还有token，则需要在bp-freq里面更新对应pair的计数
                    if pos > 0:
                        # 增加新pair的计数
                        new_pre_pair = (pre_token[pos - 1], new_bytes)
                        if new_pre_pair not in self.bp_freq:
                            self.bp_freq[new_pre_pair] = [0, set()]
                        self.bp_freq[new_pre_pair][0] += count               # 增加次数
                        self.bp_freq[new_pre_pair][1].add(have_pair_idx)     # 记录当前pre-token的idx
                        # 减去旧pair的计数
                        old_pre_pair = (pre_token[pos - 1], pair[0])
                        self.bp_freq[old_pre_pair][0] -= count
                        # self.bp_freq[old_pre_pair][1].discard(have_pair_idx)
                        if self.bp_freq[old_pre_pair][0] <= 0:
                            self.bp_freq.pop(old_pre_pair)
                    # 若后面还有token，仍需减去旧pair，处理新pair
                    if pos + 2 < len(pre_token):
                        # 增加新pair的计数
                        new_next_pair = (new_bytes, pre_token[pos + 2])
                        if new_next_pair not in self.bp_freq:
                            self.bp_freq[new_next_pair] = [0, set()]
                        self.bp_freq[new_next_pair][0] += count               # 增加次数
                        self.bp_freq[new_next_pair][1].add(have_pair_idx)     # 记录当前pre-token的idx
                        # 减去旧pair的计数
                        old_next_pair = (pair[1], pre_token[pos + 2])
                        self.bp_freq[old_next_pair][0] -= count
                        # self.bp_freq[old_next_pair][1].discard(have_pair_idx)
                        if self.bp_freq[old_next_pair][0] <= 0:
                            self.bp_freq.pop(old_next_pair)
                    # 处理当前位置的两个token
                    pre_token = pre_token[:pos] + (new_bytes,) + pre_token[pos + 2:]

                    # 更新原token计数
                    self.bp_freq[pair][0] -= count
                    if self.bp_freq[pair][0] <= 0:
                        self.bp_freq.pop(pair)
                pos += 1
            self.pre_tokens[have_pair_idx][0] = pre_token

    @staticmethod
    def count_pre_tokenize(
        file_path: str,
        special_tokens: List[str] | str,
        num_processor: int = 5,
    ) -> Dict[int, Tuple[Tuple[bytes], int]]:

        if isinstance(special_tokens, List):
            concat_special_tokens = "|".join(special_tokens)
        else:
            concat_special_tokens = special_tokens
    
        num_chunks = num_processor * 10
        boundaries = find_chunk_boundaries(file_path, num_chunks, b"<|endoftext|>")

        # 设置每个cpu要处理的文件边界
        tasks = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]
        
        # 固定变量只剩下边界
        worker = partial(
            BPETokenizerTrainer.process_chunk, 
            file_path, 
            concat_special_tokens
        )

        global_counter = Counter()
        try:
            with Pool(processes=num_processor) as pool:
                for counter in tqdm(
                    pool.imap_unordered(worker, tasks),
                    total=len(tasks),
                    desc="Spliting corpus",
                ):
                    global_counter.update(counter)
        except KeyboardInterrupt:
            print("正在清理进程池。。。")
            pool.terminate()
            pool.join()
            print("清理完毕")
        
        pre_tokens = {}
        for i, (pre_token, count) in enumerate(global_counter.items()):
            pre_tokens[i] = [pre_token, count]
        
        return pre_tokens

    @staticmethod
    def process_chunk(
        file_path: str,
        concat_special_tokens: str,
        boundary_pair: Tuple[int, int],
    ) -> Counter:
        start, end = boundary_pair
        local_counter = Counter()

        with open(file_path, "rb") as f:
            # 切块并修改换行符，测试用例里面是Linux的\n，
            # 但windows默认用的是\r\n
            f.seek(start)
            chunk = f.read(end - start).decode(encoding="utf-8", errors="ignore")
            chunk = chunk.replace("\r\n", "\n") 
            # 对文本进行预分词
            splited_chunk = re.split(concat_special_tokens, chunk)
            for seg in splited_chunk:
                pre_token_iter = Tokenizer.pre_tokenize_itera(seg)
                for pre_token in pre_token_iter:
                    local_counter[pre_token] += 1

        return local_counter

    def save(self, save_path: str | None = None) -> None:
        if not save_path:
            save_path = str(Path(__file__).parent / "tokenizer_params.bin")
        if not save_path.lower().endswith(".bin"):
            save_path += ".bin"

        # # 直接二进制存
        # with open(save_path, "wb") as f:
        #     save_data = {
        #         "vocab": self.vocab,
        #         "merges": self.merges,
        #     }
        #     pickle.dump(save_data, f)
        #     f.close()

        # 自定义结构体存
        # 每个数据所占的字节长度
        size_to_fmt = { 1: "B", 2: "H", 4: "I", 8: "Q" }
        id_bytes = next(b for b in [1, 2, 4, 8] if self.vocab_size <= 256 ** b)
        fmt = size_to_fmt[id_bytes]
        with open(save_path, "wb") as f:
            # 保存每个id用的字节数
            f.write(struct.pack("B", id_bytes))

            ''' 
                保存vocab，保存格式为
                    vocab词表大小N: 4字节
                    词表ID(key): 4 字节
                    ID对应字节长度n: 4 字节
                    ID对应字节: n 字节
            '''
            f.write(struct.pack(fmt, self.vocab_size))
            for k, v in self.vocab.items():
                num_bytes = len(v)
                f.write(struct.pack(2 * fmt, k, num_bytes))
                f.write(v)

            ''' 
                保存merges
                目前merges存的是字节对，需要先转换成ID对再存，读取的时候再转换回来
                merges保存格式为
                    merges列表大小N: 4 字节
                    merges参数: ID1 + ID2: (4 + 4) * N 字节
                总长度为 8N + 4 字节
            '''
            # 保存列表长度
            size_merges = len(self.merges)
            f.write(struct.pack(fmt, size_merges))
            # 字节对转化成ID对
            for b1, b2 in self.merges:
                id1, id2 = self.vocab_to_index[b1], self.vocab_to_index[b2]
                f.write(struct.pack(2 * fmt, id1, id2))
            
            f.close()


class BPETokenizer(Tokenizer):
    def __init__(
        self, 
        vocab: Dict[int, bytes] = {},
        merges: List[Tuple[bytes, bytes]] = [],
        special_tokens: List[str] | str = [],
    ) -> None:
        super().__init__()

        # 加入默认的特殊tokens
        if isinstance(special_tokens, List):
            if "<|endoftext|>" not in special_tokens:
                special_tokens.append("<|endoftext|>")
            special_tokens = sorted(special_tokens, key=len, reverse=True)
        self.special_tokens = special_tokens
        
        # 初始化词汇表和合并列表
        self.vocab = vocab
        self.vocab_to_index = {v: k for k, v in self.vocab.items()}
        self.merges = merges

    @classmethod
    def from_files(
        cls, 
        params_path: str, 
        special_tokens: List[str] | None = None
    ) -> Self:

        size_to_fmt = { 1: "B", 2: "H", 4: "I", 8: "Q" }
        with open(params_path, "rb") as f:
            size = f.read(1)[0]
            fmt = size_to_fmt[size]

            # 读取 vocab
            # 不加[0]读出来就是一个(ans,)形式的列表
            vocab_size = struct.unpack(fmt, f.read(size))[0]
            vocab = {}
            for _ in range(vocab_size):
                token_id, token_size = struct.unpack(2 * fmt, f.read(2 * size))  # 数字形式
                token = f.read(token_size)  # 字节形式
                assert token_id not in vocab, f"Token id {token_id} appears repeatedly in vocab."
                vocab[token_id] = token

            # 读取 merges
            merges_size = struct.unpack(fmt, f.read(size))[0]
            merges = []
            for _ in range(merges_size):
                id1, id2 = struct.unpack(2 * fmt, f.read(2 * size))
                token1, token2 = vocab[id1], vocab[id2]
                merges.append((token1, token2))

            f.close()

        # 加入默认的特殊tokens
        if special_tokens and isinstance(special_tokens, List):
            if "<|endoftext|>" not in special_tokens:
                special_tokens.append("<|endoftext|>")
            special_tokens = sorted(special_tokens, key=len, reverse=True)

        return cls(vocab, merges, special_tokens)

    def merge(
        self,
        pre_tokens_to_token: Dict[Tuple[bytes], Tuple[bytes]]
    ) -> None:
        # 统计要合并的byte-pair在哪些pre-token里面出现过
        bp_to_pre_tokens: set = {}
        for pre_token in pre_tokens_to_token.keys():
            for bp in zip(pre_token[:-1], pre_token[1:]):
                if bp not in bp_to_pre_tokens:
                    bp_to_pre_tokens[bp] = {pre_token}
                else:
                    bp_to_pre_tokens[bp].add(pre_token)

        for merge in self.merges:
            if merge not in bp_to_pre_tokens:
                continue
            merged_token = merge[0] + merge[1]
            for pre_token in bp_to_pre_tokens[merge]:
                new_token = pre_tokens_to_token[pre_token]
                # 根据merge选项合并pre-token中的对应byte-pair
                # 同时更新bp-to-pre-tokens
                pos = 0
                while pos < len(new_token) - 1:
                    bp = (new_token[pos], new_token[pos + 1])
                    if bp == merge:
                        # 检查前方有没有要合并的token
                        if pos > 0:
                            bp_previous = (new_token[pos - 1], merged_token)
                            if bp_previous not in bp_to_pre_tokens:
                                bp_to_pre_tokens[bp_previous] = {pre_token}
                            else:
                                bp_to_pre_tokens[bp_previous].add(pre_token)
                        # 检查后方有没有要合并的token
                        if pos + 2 < len(new_token):
                            bp_next = (merged_token, new_token[pos + 2])
                            if bp_next not in bp_to_pre_tokens:
                                bp_to_pre_tokens[bp_next] = {pre_token}
                            else:
                                bp_to_pre_tokens[bp_next].add(pre_token)
                        # 合并找到的merge bp对
                        new_token = new_token[:pos] + (merged_token, ) + new_token[pos + 2:]
                    pos += 1
                pre_tokens_to_token[pre_token] = new_token

    def encode(self, text: str) -> List[int]:
        (
            pre_tokens_to_token,    # 统计有多少种pre-token（字典）
            pre_tokens_seq,         # pre-token序列
            special_tokens_seq      # 特殊token序列
        ) = self.pre_tokenize(text, self.special_tokens)

        # 计算pre-tokens到token的映射
        self.merge(pre_tokens_to_token)

        pos = 0
        token_id_list = []
        for pre_token in pre_tokens_seq:
            # 特殊token整个处理
            if (
                special_tokens_seq 
                and pos < len(special_tokens_seq)
                and pre_token == special_tokens_seq[pos]
            ):
                # str形式转换成bytes形式
                pre_token = bytes(pre_token.encode("utf-8"))
                token_id = self.vocab_to_index[pre_token]
                token_id_list.append(token_id)
                pos += 1
                continue
            # 正常token要根据合并情况处理
            tokens = pre_tokens_to_token[pre_token]
            for token in tokens:
                token_id = self.vocab_to_index[token]
                token_id_list.append(token_id)
        return token_id_list

    def encode_iterable(self, file: IO) -> Iterable:
        """
        需要保证分块pre-token不会截断正常pre-token和特殊token
        可以在块内检测有无特殊token，有则截断到该token处
        没有则在pre-token后舍弃最后一个pre-token，避免产生截断
        """

        chunk_size = 64

        file_pos = 0
        while True:
            chunk = file.read(chunk_size)

            # 检查是否是最后一个块
            if not chunk:
                break
            
            # 如果是最后一个块，跳过截断判断
            is_last_chunk = len(chunk) < chunk_size

            if not is_last_chunk:
                original_chunk = chunk
                # 查看是否有特殊tokens
                if self.special_tokens:
                    concat_special_token = "|".join([
                        re.escape(token) for token in self.special_tokens
                    ])
                    # 检测chunk最后一段是否存在特殊token，前面存在的就不管了，encode可以区分
                    max_sp_token = len(max(self.special_tokens, key=len))
                    end_chunk = chunk[-2 * max_sp_token:]
                    matches = list(re.finditer(concat_special_token, end_chunk))
                    # 若检测到，则取第一个起始位置作为分割位置
                    if len(matches) > 0:
                        split_pos = matches[0].span()[0]
                        chunk = chunk[:split_pos]
                    # 若没检测到，则检查最后一个pre-token的长度
                    else:
                        # 避免最后一个是被截断的特殊token
                        chunk = chunk[:-max_sp_token]
                        pre_tokens = re.findall(PAT, chunk)
                        if pre_tokens:
                            last_token_len = len(pre_tokens[-1])
                            chunk = chunk[:-last_token_len]
                # 若没有特殊tokens
                else:
                    pre_tokens = re.findall(PAT, chunk)
                    if pre_tokens:
                        last_token_len = len(pre_tokens[-1])
                        chunk = chunk[:-last_token_len]
                
                if chunk == "":
                    chunk = original_chunk
            # 处理完chunk之后进行encode
            chunk_len = len(chunk)

            # 文件指针移动到截断后长度的位置
            # 错误示例，seek输入是字节数，不是字符数，得转换一下，或者用read
            # file.seek(file_pos)
            # file.read(chunk_len)
            # a = file.tell()
            # file_pos += chunk_len
            # file.seek(file_pos)
            # b = file.tell()
            # print(a, b)

            # 获取字符串截断后对应的文件指针位置
            file.seek(file_pos)
            file.read(chunk_len)
            file_pos = file.tell()
            file.seek(file_pos)

            token_id_list = self.encode(chunk)
            for token_id in token_id_list:
                yield token_id


    def decode(self, tokens: List[int]) -> str:
        decode_bytes = b""
        for token in tokens:
            decode_bytes += self.vocab[token]
        return decode_bytes.decode(encoding="utf-8", errors="ignore")


def test_encode_iterable():
    special_tokens = ["<|endoftext|>", "<|endoftext|><|endoftext|>"]
    
    bpe = BPETokenizer.from_files(
        str(Path(__file__).parent / "tokenizer_params.bin"), 
        special_tokens
    )

    origin_text = """
    
Once upon a time there was a little boy named Ben. Ben loved to explore the world around him. He saw many amazing things, like beautiful vases that were on display in a store. One day, Ben was walking through the store when he came across a very special vase. When Ben saw it he was amazed!
He said, “Wow, that is a really amazing vase! Can I buy it?”
The shopkeeper smiled and said, “Of course you can. You can take it home and show all your friends how amazing it is!”
So Ben took the vase home and he was so proud of it! He called his friends over and showed them the amazing vase. All his friends thought the vase was beautiful and couldn't believe how lucky Ben was.
And that's how Ben found an amazing vase in the store!
<|endoftext|>
Once upon a time, there was a reliable otter named Ollie. He lived in a river with his family. They all loved to play and swim together.
One day, Ollie's mom said, "Ollie, hurry and get some fish for dinner!" Ollie swam fast to catch fish. He saw his friend, the duck. "Hi, Ollie!" said the duck. "Hi, duck!" said Ollie. "I need to hurry and catch fish for my family."
While Ollie was catching fish, he found a big shiny stone. He thought, "This is not a fish, but it is so pretty!" Ollie took the shiny stone home to show his family. They all looked at the shiny stone and smiled. The shiny stone made everyone happy, and they forgot about the fish for dinner.
<|endoftext|>
One day, a little boy named Tim went to the park. He saw a big tiger. The tiger was not mean, but very easy to play with. Tim and the tiger played all day. They had lots of fun.
Then, something unexpected happened. The tiger started to shake. Tim was scared. He did not know what was going on. But then, the tiger turned into a nice dog. Tim was very surprised.
Tim and the dog played together now. They were very happy. The dog was easy to play with too. At the end of the day, Tim went home with his new friend.
<|endoftext|>

Once upon a time there was a friendly little boy called Bob. Bob loved to pick flowers and look for birds. One day he decided to go outside with his friends to pick some more flowers.
He suddenly noticed something weird on the ground. It was a big, green thumb! It was so big, Bob had never seen one before. Bob curiously leaned in to take a better look. He told his friends: "look everyone, I picked up this big thumb! What do we do with it?"
His friends were very excited. They told him to pick it up and take it home to show his family. So Bob carefully picked up the friendly thumb and carried it back home. When he arrived, Bob happily showed the thumb to his family. His dad was amazed and hugged Bob to show his appreciation.
From that day on Bob always kept the big, friendly thumb with him as a reminder that special things can be found anywhere.
<|endoftext|>
Once upon a time, in a small house, there lived a little girl named Lucy. Lucy loved the color orange. She had an orange dress, an orange ball, and even an orange cat. One day, Lucy met a new friend. This friend was not like other friends. It was a spirit. The spirit was very nice and liked to play with Lucy.
One day, Lucy and the spirit were playing with her orange ball. They were having so much fun. Then, Lucy's mom called her for dinner. Lucy said to the spirit, "I have to go eat now. Will you play with me later?" The spirit nodded and smiled.
At dinner, Lucy told her mom about the spirit. But her mom did not believe her. She said, "Spirits are not real, Lucy. You have a big imagination." Lucy felt sad that her mom did not believe her. After dinner, she went back to play with the spirit. They played with the orange ball and had lots of fun. Lucy knew that even if others ignore her friend, the spirit was real and they could play together.
<|endoftext|>

    """
    token_id_list = bpe.encode(origin_text)
    # print(token_id_list)

    decode_text = bpe.decode(token_id_list)
    # print("origin text: ", origin_text)
    # print("decode text: ", decode_text)
    
    test_path = Path(__file__).parent / "text.txt"
    with open(test_path, 'w') as file:
        file.write(origin_text)
    
    ids = []
    with open(test_path, 'r') as file:
        token_iter = bpe.encode_iterable(file)
        for token_id in token_iter:
            ids.append(token_id)
    decode_itera = bpe.decode(ids)

    # print("decode text: ", decode_itera)
    from rich.console import Console
    from rich.table import Table
    console = Console()
    table = Table(title="BPE Tokenizer 差异对比")
    
    table.add_column("位置", justify="right")
    table.add_column("预期 (Expected)", style="green")
    table.add_column("实际 (Actual)", style="red")
    table.add_column("差异说明", style="bold yellow")
    max_len = max(len(decode_itera), len(origin_text))
    for i in range(max_len):
        exp = origin_text[i] if i < len(origin_text) else "-"
        act = decode_itera[i] if i < len(decode_itera) else "-"
        
        # 转换成字符看
        exp_char = f"'{exp}" if exp != "-" else "-"
        act_char = f"'{act}" if act != "-" else "-"
        
        if exp != act:
            table.add_row(str(i), exp_char, act_char, "❌ 不匹配" if act != "-" and exp != "-" else "➕ 多出")
            # 如果只想看报错点附近，可以在这里加个计数器，只打前后 5 行
    # console.print(table)
    # print(decode_itera)
    assert decode_itera == origin_text


if __name__ == "__main__":
    
    special_tokens = ["<|endoftext|>", "<|endoftext|><|endoftext|>"]
    input_path = Path(__file__).parent.parent / "data" / "TinyStoriesV2-GPT4-valid.txt"
    
    # trainer = BPETokenizerTrainer(
    #     file_path = input_path, 
    #     vocab_size=10000, 
    #     special_tokens=special_tokens
    # )
    # trainer.train_bpe()
    # trainer.save()

    # bpe = BPETokenizer(trainer.vocab, trainer.merges, trainer.special_tokens)
    bpe = BPETokenizer.from_files(
        str(Path(__file__).parent / "tokenizer_params.bin"), 
        special_tokens
    )

    origin_text = "Hello world! <|endoftext|><|endoftext|> Hello <|endoftext|> world! <|endoftext|>"

    token_id_list = bpe.encode(origin_text)
    # print(token_id_list)

    decode_text = bpe.decode(token_id_list)
    # print("origin text: ", origin_text)
    # print("decode text: ", decode_text)
    
    test_path = Path(__file__).parent / "text.txt"
    with open(test_path, 'w') as file:
        file.write(origin_text)
    
    ids = []
    with open(test_path, 'r') as file:
        token_iter = bpe.encode_iterable(file)
        for token_id in token_iter:
            ids.append(token_id)
    decode_itera = bpe.decode(ids)

    print("origin text: ", origin_text)
    print("decode text: ", decode_text)
    print("decode iter: ", decode_itera)