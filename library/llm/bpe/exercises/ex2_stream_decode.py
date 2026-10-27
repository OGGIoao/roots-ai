"""
练习 2 (② 接口): 流式解码器

LLM 生成时一次吐一个 token。如果每来一个 token 就单独 decode 一次再打印, 汉字和 emoji 会变成一串 "�":
一个字符的 UTF-8 字节可能分散在好几个 token 里 (讲义第 8.2 节、第 11.2 节)。

实现 StreamDecoder: 一个一个地接收 token ID, 每次只返回**已经完整**的字符, 不完整的字节先攒着。

要求 (测试逐条检查):
  1. 把所有 feed() 的返回值和最后的 flush() 拼起来, 等于一次性 decode 全部 ID 的结果;
  2. 对合法的 UTF-8 流, feed() 从不返回 "�";
  3. 不拖延: 每次 feed() 之后, 已返回的文本恰好是"目前收到的字节里最长的完整前缀";
  4. flush() 输出剩下的字节; 结尾不完整时按 errors="replace" 处理 (与 bytes.decode 一致);
  5. 不许用 codecs 的增量解码器, 自己按 UTF-8 的规则判断一个字符有没有收完 (讲义 16.2 节的字节模板)。

验收:
    cd library/llm/bpe/exercises
    python -m pytest test_ex2.py -q
"""


class StreamDecoder:
    def __init__(self, vocab: dict[int, bytes]):
        """vocab: token ID → 这个 token 代表的字节 (就是 BPETokenizer.vocab)。"""
        raise NotImplementedError

    def feed(self, token_id: int) -> str:
        """收下一个 token, 返回因此新变得完整的文本 (可能是空串)。"""
        raise NotImplementedError

    def flush(self) -> str:
        """生成结束: 返回剩下没输出的部分, 并清空内部状态。"""
        raise NotImplementedError
