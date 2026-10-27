# 两组“生成母模型—决策模型”配对

核查日期：2026-10-24。只读取公开模型卡和小型元数据；未下载权重、加载模型、训练或运行实验。

用户最新澄清：优先寻找**决策模型实际声明的直接生成母模型**。这里 Base 指训练起点／母模型，不强制要求检查点名称为 `-Base` 或只经过预训练。这修正了上一版把“Base”限定为“仅预训练”的理解。

## 主设置

| 规模 | 生成模型（母模型） | 决策模型（衍生模型） | 公开依据 |
|---|---|---|---|
| 小组，9B 级 | `Qwen/Qwen3.5-9B` | `vllm-sr/d3-flash@v3.1.0` | 模型卡 YAML `base_model: Qwen/Qwen3.5-9B`；HF revision API 同样声明 |
| 大组，27B 级 | `Qwen/Qwen3.8-27B` | `vllm-sr/d3@v3.1.0` | 模型卡 YAML `base_model: Qwen/Qwen3.8-27B`；HF revision API 同样声明 |

9B 的直接配对不是 `Qwen3.5-9B-Base`：后者是更早的预训练祖先，而 d3-flash 声明的直接母模型是后训练的 Qwen3.5-9B。两组生成模型均支持官方原生 thinking，这与用户“两者都开 thinking”的要求相容。

模型卡：

- [d3-flash 当前模型卡：Built on Qwen/Qwen3.5-9B](https://huggingface.co/vllm-sr/d3-flash)
- [d3-flash 固定版本的 README](https://huggingface.co/vllm-sr/d3-flash/blob/521525effa79469d899a570726547dd7f00da3af/README.md)
- [d3 v3.1.0 的 README：base_model 为 Qwen/Qwen3.8-27B](https://huggingface.co/vllm-sr/d3/blob/v3.1.0/README.md)
- [9B 生成母模型](https://huggingface.co/Qwen/Qwen3.5-9B)
- [27B 生成母模型](https://huggingface.co/Qwen/Qwen3.8-27B)

## 已读取并记录的 revision

| 角色 | repo | 固定 SHA |
|---|---|---|
| D_S | vllm-sr/d3-flash，v3.1.0 | `521525effa79469d899a570726547dd7f00da3af` |
| D_L | vllm-sr/d3，v3.1.0 | `cbff3a3cb0b074b6585fe9f747b9f9f5557a0e75` |
| G_S | Qwen/Qwen3.5-9B | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| G_L | Qwen/Qwen3.8-27B | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |

原始 API 响应和两份固定版本 README 存于 `model-provenance/`。这些 SHA 固定的是我们计划使用的公开版本；模型作者没有在所查材料中披露训练初始化母模型的精确 commit，因此不能把这张表声称为“已证明完全相同的初始化权重”。证据级别是作者声明的直接来源 repo。

决策模型卡将两种决策模型归为约 9B/27B 级，实际报告参数分别约 8.39B/26.09B。生成和决策输出结构有差异；我们的配对依据是声明的母模型关系，不要求参数总数逐项相等，也不声称只替换一个头就可解释全部差异。

## 固定部署身份

两组都自行加载固定 revision 的权重与原生模型代码。[官方公共 API 文档](https://decision.vllm-sr.ai/docs#models) 在 2026-10-24 的核查记录中将公共名称 `vllm-sr/d3` 指向 `d3-turbo`；公共 alias 因此不能证明加载了本研究指定的原版 d3。运行器应核对本地 snapshot SHA、原生 manifest 和加载记录。

## 双向监督

- 大监督小：D_L（d3 27B 级）选择 G_S（Qwen3.5-9B thinking）的候选补丁。
- 小监督大：D_S（d3-flash 9B 级）选择 G_L（Qwen3.8-27B thinking）的候选补丁。
- 同源同规模对照：D_S 选择 G_S；D_L 选择 G_L。

仍采用 1–2 题联调、固定 50 题探索性评测、完整轨迹 Best-of-3、共享候选池、单轨迹/随机/Oracle 对照。主工作量保持 300 条完整轨迹和最多 200 次选择。

## 解释边界

这组设置控制了每个规模上“生成器与决策器是否来自不相关的母模型”这一问题。但两个规模跨代际，决策模型也经过专门训练并改变读出机制；不能把效果差异解释成模型大小的纯因果效应。模型卡来源关系不等于完整训练谱系审计，训练数据、精确初始化 revision 等未披露项应保留为未知。
