# 知识库 · 多模态

> 图像、声音怎么变成模型能处理的 token: 视觉、语音、生成模型基础、多模态大模型。总目录见 [library/README.md](../README.md#multimodal)。

**为什么单独成一个领域**: 经典视觉和语音本身就是两大门类 (小车 v0.6 找红杯子、v1.0 语音指挥都要用); 生成模型基础 (VAE、扩散、流匹配) 被三个领域共用 —— 具身领域的扩散策略和视频世界模型、LLM 领域的扩散语言模型都链接到这里。

**基座线**: 基座线 [roots-lm](../../base/README.md) 的阶段 4 "长眼睛和耳朵" 用到 M4 和 M2 的 audio-codecs。

**已有原稿**: 姊妹仓库 roots-modeling 搬家前写过一本 CIFAR-10 "图片怎样变成可计算的东西" (像素与通道、向量表示、平方距离、类别模板、线性得分, 带手写练习和折叠参考解), 保存在它的标签 [`pre-restructure`](https://github.com/OGGIoao/roots-modeling/tree/pre-restructure/深度学习/CIFAR-10图片分类) 里 (roots-modeling 暂未公开)。写 image-basics 单元时改成本仓库的格式收进来。

## 单元

<!-- catalog:start -->
| 段 | 单元 | 讲什么 | 状态 | 用到它的地方 |
|---|---|---|---|---|
| M0 总览 | multimodal-map | 理解与生成两个方向的全景 | ⬜ |  |
| M1 视觉 | image-basics | 像素、颜色、卷积 | ⬜ |  |
|  | cnn | 卷积网络、ResNet | ⬜ |  |
|  | detection | 目标检测: 小设备上的轻量检测器 | ⬜ | v0.6 |
|  | vit | Vision Transformer: 图片切块当 token | ⬜ | 基座 |
|  | self-supervised-vision | CLIP、DINO: 不靠标注学图像表示 | ⬜ |  |
| M2 音频与语音 | audio-basics | 采样、频谱、梅尔谱 | ⬜ | v1.0 |
|  | keyword-spotting | 唤醒词识别 | ⬜ | v1.0 |
|  | asr | 语音识别 | ⬜ | v1.0 |
|  | tts | 语音合成 | ⬜ | v1.0 |
|  | audio-codecs | 把声音变成离散 token 🔥 | ⬜ | 基座 |
| M3 生成模型基础 | vae | 变分自编码器 | ⬜ |  |
|  | diffusion | DDPM、扩散原理、流匹配 | ⬜ |  |
|  | image-generation | 潜空间扩散、DiT | ⬜ |  |
|  | video-generation | 视频生成, 通往视频世界模型 | ⬜ |  |
| M4 多模态大模型 | vision-language-alignment | CLIP 式对齐 | ⬜ | 基座 |
|  | vlm | 视觉编码器 + 投影层 + 语言模型 | ⬜ | 基座 |
|  | native-multimodal | 早期融合、不用单独编码器 🔥 | ⬜ | 基座 |
|  | omni-models | 能听能说、实时全双工的全能模型 🔥 | ⬜ | 基座 |
|  | multimodal-eval | 多模态怎么评测 | ⬜ |  |
<!-- catalog:end -->

**前置**: 基础领域的 linear-algebra、pytorch-basics; M4 还需要 LLM 领域的 [minigpt](../llm/minigpt/TUTORIAL.md)。
