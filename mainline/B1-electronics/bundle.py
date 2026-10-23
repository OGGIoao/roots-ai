"""
把 b1core.py 和一个案例程序拼成单个 main.py, 方便粘贴到 Wokwi 网页版。

Wokwi 网页版里新建文件要手动操作; 拼成一个文件后, 只需要把 main.py 整个替换掉。
拼接方式: b1core.py 原样放在前面, 案例程序里的 "from b1core import ..." 一行去掉 (名字已经在前面定义了)。

用法:
    python bundle.py firmware/case3_pwm.py              # 打印到屏幕
    python bundle.py firmware/case3_pwm.py -o main.py   # 写到文件

真板上不需要这一步: 把 b1core.py 和案例程序分别拷到板子上即可 (见 README)。
"""

import argparse
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def bundle(case_path):
    core = open(os.path.join(HERE, "b1core.py"), encoding="utf-8").read()
    case = open(case_path, encoding="utf-8").read()
    kept = [line for line in case.splitlines() if not line.startswith("from b1core import")]
    return core.rstrip() + "\n\n\n# " + "=" * 20 + " " + os.path.basename(case_path) + " " + "=" * 20 + "\n\n" + "\n".join(kept) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("case")
    ap.add_argument("-o", "--output")
    args = ap.parse_args()
    text = bundle(args.case)
    if args.output:
        with open(args.output, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
    else:
        print(text)
