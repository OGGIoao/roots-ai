"""
练习 3 (③ 白纸): 按住呼吸、松开熄灭的 LED

只给验收标准, 怎么实现自己决定。不要照抄 ../b1core.py; 写完、测试通过之后再去对照。

要实现的函数:

    make_breathing() -> obj
    obj.update(pressed: bool, now_ms: int) -> int    # 返回此刻的 duty_u16 (0~65535 的整数)

验收 (python -m pytest test_ex3.py -q):
  1. 按住不放时 LED "呼吸": 占空比周期性地变亮变暗, 一个来回 1.5~2.5 秒;
     每个来回里最暗 ≤ 1% (655), 最亮 ≥ 99% (64880);
  2. 变化是平滑的: 相邻两毫秒之间, "看起来的亮度" (duty/65535 的 1/2.2 次方) 变化不超过 0.01;
  3. 松开后 1 秒内熄灭 (duty == 0), 熄灭过程中只变暗不变亮, 之后一直保持 0;
  4. 再次按下时重新开始呼吸;
  5. 任何时候返回值都是 0~65535 的整数。

上芯片 (选做, 不计入测试): 按钮接 GPIO 5、LED 接 GPIO 4, 用 machine.PWM(Pin(4), freq=1000) 驱动,
按钮读数先经过练习 2 的消抖器。用 Wokwi 的逻辑分析仪确认 PWM 频率是 1 kHz ± 1%。

讲义对应: 第 3 节 (PWM、gamma) 和第 5 节 (非阻塞)。
"""


def make_breathing():
    raise NotImplementedError
