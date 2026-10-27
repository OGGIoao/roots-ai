"""
练习 3 (③ 白纸): MLA 的吸收式解码

只给验收标准, 实现方式自己设计。讲义第 5.5 节给了推导, 但不要照着 ../decode.py 抄,
写完、测试通过之后再去对照。

要实现的函数:

    absorbed_step(attn, h, cos, sin, cache) -> out

    attn   一个 variants.MultiHeadLatentAttention 模块 (读它的权重, 不要修改它)
    h      (B, 1, C)  新 token 经过 norm1 之后的输入
    cos    (1, d_rope/2)  这个位置的 RoPE 表; sin 同理
    cache  一个 dict, 第一次调用时为空; 你决定往里存什么
    out    (B, 1, C)  这一层注意力的输出 (已经过输出投影 wo)

验收 (python -m pytest test_ex3.py -q):
    1. 逐 token 解码 40 个位置, 每个位置的 logits 与整段前向一致 (误差 < 1e-4);
    2. cache 里每个位置只存 kv_rank + d_rope 个数, 不能缓存解压后的 K、V;
    3. 解码过程中不能把历史位置的潜向量解压成每个头的 K 或 V
       (测试会把 attn.w_ukv 替换成"只允许对新 token 调用"的版本来检查)。
"""


def absorbed_step(attn, h, cos, sin, cache):
    raise NotImplementedError
