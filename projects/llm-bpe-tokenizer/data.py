"""
data.py — 训练语料获取与准备

支持两种模式：
  1. 自动下载 TinyShakespeare（约 1MB，推荐）
  2. 内置 sample 文本（无网络时的备用）

教练注：
  用 TinyShakespeare 训练 tokenizer 是经典入门选择。
  - 文件小（~1MB），下载快
  - 英文，内容多样（对话、诗歌、散文）
  - 和 Karpathy 的 nanoGPT/makemore 共用同一份数据
  - 后续 L3 (mini-GPT) 也会用它训练语言模型
"""

# 语料的定位 / 下载逻辑全仓库只有一份 (common/corpus.py), 本项目是语料的所有者:
# 文件就入库在本目录的 data/ 下。
try:
    from common.corpus import load_shakespeare
except ModuleNotFoundError as e:  # 只拦 common 本身缺失, 不吞其它依赖的报错
    if e.name != "common":
        raise
    raise SystemExit(
        "找不到共享包 `common`。请在仓库根目录执行一次:\n"
        "    pip install -e . --no-deps\n"
        "(详见根目录 README 的「安装」一节)"
    ) from None

# 内置 sample 文本：当网络不可用时使用
# 包含英文、数字、标点，足以演示 BPE 基本行为
_SAMPLE_TEXT = """\
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die—to sleep,
No more; and by a sleep to say we end
The heartache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep, perchance to dream—ay, there's the rub:
For in that sleep of death what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause. There's the respect
That makes calamity of so long life.

FIRST CITIZEN:
Before we proceed any further, hear me speak.

ALL:
Speak, speak.

FIRST CITIZEN:
You are all resolved rather to die than to famish?

ALL:
Resolved. resolved.

FIRST CITIZEN:
First, you know Caius Marcius is chief enemy to the people.

ALL:
We know't, we know't.

FIRST CITIZEN:
Let us kill him, and we'll have corn at our own price.
Is't a verdict?

ALL:
No more talking on't; let it be done: away, away!

SECOND CITIZEN:
One word, good citizens.

FIRST CITIZEN:
We are accounted poor citizens, the patricians good.
What authority surfeits on would relieve us: if they
would yield us but the superfluity, while it were
wholesome, we might guess they relieved us humanely;
but they think we are too dear: the leanness that
afflicts us, the object of our misery, is as an
inventory to particularise their abundance; our
sufferance is a gain to them Let us revenge this with
our pikes, ere we become rakes: for the gods know I
speak this in hunger for bread, not in thirst for revenge.

SECOND CITIZEN:
Would you proceed especially against Caius Marcius?

ALL:
Against him first: he's a very dog to the commonalty.

SECOND CITIZEN:
Consider you what services he has done for his country?

FIRST CITIZEN:
Very well; and could be content to give him good
report for't, but that he pays himself with being proud.

SECOND CITIZEN:
Nay, but speak not maliciously.

FIRST CITIZEN:
I say unto you, what he hath done famously, he did
it to that end: though soft-conscienced men can be
content to say it was for his country, he did it to
please his mother and to be partly proud; which he
is, even till the altitude of his virtue.
""" * 20  # 重复 20 次，模拟更大的语料


def get_corpus(source: str = "shakespeare", verbose: bool = True) -> str:
    """
    获取训练语料。

    Args:
        source: "shakespeare" 或 "sample"
            - "shakespeare"：入库的 TinyShakespeare（~1MB）；缺失才下载，
              下载失败会**抛出 RuntimeError**，不会悄悄换成别的语料
            - "sample"：内置小语料，需要时由调用方显式选择（如离线演示）
        verbose: 是否打印进度信息

    Returns:
        语料字符串
    """
    if source == "shakespeare":
        return _load_shakespeare(verbose=verbose)
    elif source == "sample":
        if verbose:
            print(f"[数据] 使用内置 sample 语料，{len(_SAMPLE_TEXT):,} 字符")
        return _SAMPLE_TEXT
    else:
        raise ValueError(f"未知 source='{source}'，请选择 'shakespeare' 或 'sample'")


def get_train_val_split(
    text: str, val_ratio: float = 0.1
) -> tuple[str, str]:
    """
    把语料按比例切分为训练集和验证集。

    注意：不能随机 shuffle！文本是有顺序的，
    我们取前 90% 为训练集，后 10% 为验证集。

    Args:
        text: 完整语料
        val_ratio: 验证集比例（默认 10%）

    Returns:
        (train_text, val_text)
    """
    split_idx = int(len(text) * (1 - val_ratio))
    return text[:split_idx], text[split_idx:]


# ─── 内部辅助函数 ─────────────────────────────────────────────────────────────

def _load_shakespeare(verbose: bool) -> str:
    """读取 TinyShakespeare (缺失时下载; 失败则抛错, 不做静默降级)。"""
    text = load_shakespeare()
    if verbose:
        print(f"[数据] 语料大小：{len(text):,} 字符 / {len(text.encode('utf-8')):,} 字节")
    return text
