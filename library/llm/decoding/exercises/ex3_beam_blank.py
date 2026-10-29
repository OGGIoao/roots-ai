"""
练习 3 (③ 白纸): 束搜索

只看下面的规则, 从零写出束搜索 (讲义第 9 节):

    beam_search(step, prompt_ids, beam_width, max_new_tokens, stop_ids=()) -> (新生成的 ids 列表, 总分)

  - 一条路的总分 = 每一步所选 token 的 log softmax 概率之和;
  - 每一步, 把每条没结束的路往下展开一个 token (词表里每个 token 都试一遍);
    以停止 token 结尾的路算结束了, 不再展开, 但原样留在候选里和别人比;
  - 候选按 (总分从高到低, 再按 token 序列从小到大) 排序, 留前 beam_width 条;
  - 留下的路全部结束、或者已经走了 max_new_tokens 步时, 返回排第一的那条;
  - 不做长度归一化。

验收 (和穷举比、和贪心比、在 ToyLM 上和参考实现比):
    cd library/llm/decoding/exercises
    python -m pytest test_ex3.py -q

写完想一想: 束越宽, 总分越高, 为什么 ToyLM 上束搜索的结果照样在 "the the the" 里转圈?
"""


def beam_search(step, prompt_ids, beam_width, max_new_tokens, stop_ids=()):
    raise NotImplementedError
