# 双向决策监督的 empirical study：轨迹级 Best-of-3

本项目已提供 GPU 推理、容器内解题、共享候选池、双选择器、官方评测和汇总分析的运行代码。**本机只做 CPU 模拟测试，未下载权重、未启动 GPU 推理、未运行真实实验。**真实兼容性和显存峰值须在目标设备的 1–2 题联调中验收。

目标平台为 **Linux x86_64 + NVIDIA CUDA / BF16 GPU + Docker**。运行路径、GPU 编号、TP、缓存、容器资源和评测并发全部由配置指定，不使用当前 Windows 电脑的配置。迁移前填写 [新设备确认清单](NEW_DEVICE_CHECKLIST.md)。

## 实验设置

| 角色 | 固定检查点 | GPU 后端 |
|---|---|---|
| 小生成器 G_S | Qwen/Qwen3.5-9B | vLLM，原生 thinking 开启，可配 TP |
| 大生成器 G_L | Qwen/Qwen3.8-27B | vLLM，原生 thinking 开启，可配 TP |
| 小选择器 D_S | vllm-sr/d3-flash，v3.1.0 | 官方原生 `system_one` Choice，单 GPU BF16 |
| 大选择器 D_L | vllm-sr/d3，v3.1.0 | 官方原生 `system_one` Choice，单 GPU BF16 |

四个模型的完整 commit SHA 已固定在 [GPU 配置](configs/gpu.example.json)。D3 模型卡声明的直接生成母模型及其证据边界见 [MODEL_PAIRS.md](MODEL_PAIRS.md)。此处“Base”沿用用户澄清后的母模型含义，不是要求仅预训练版本。原版 d3 的公共 API 名称已成为 turbo 的别名，本项目自行加载固定原版权重。

先在 **2 道不属于 Verified 的 dev 题**走通流程，再在排除此前查看过的 10 题后，用固定 seed 抽取 **50 道 SWE-bench Verified**。debug 和 pilot 均固定 N=3，每个生成器每题独立生成 3 条完整轨迹；两个选择器评判完全相同的池与呈现顺序。共有大监督小、小监督大、两个同规模对照，以及 Single-first、随机选择期望值、Oracle@3。50 题最多 300 条轨迹、200 次实际选择和 300 个候选评测，失败尝试的额外开销另计。

选择器只看 issue、完整 diff、已观察命令结果摘录。两位选择器的全部决策写入并封存后，才允许官方评测。生成器原生 reasoning 与 final 分开保存，**只执行 final 中严格解析出的一个 bash 块**；测试或评测文件被改动的候选按协议无效计入主指标，不补采样。

## 1. 在新设备建立三个环境

以下命令均供目标设备执行，当前对话不会自动执行它们。建议 Python 3.11。先按目标 CUDA/驱动及各项目官方说明安装匹配的 PyTorch/vLLM 二进制；不要直接复制其他机器的 CUDA wheel。

```bash
cd /path/to/d3-bidirectional-bon-study
python3.11 -m venv .venv-generator
.venv-generator/bin/python -m pip install --upgrade pip
.venv-generator/bin/python -m pip install -e '.[generator]'

python3.11 -m venv .venv-decision
.venv-decision/bin/python -m pip install --upgrade pip
.venv-decision/bin/python -m pip install -e '.[decision]'

python3.11 -m venv .venv-evaluation
.venv-evaluation/bin/python -m pip install --upgrade pip
.venv-evaluation/bin/python -m pip install -e '.[data,evaluation]'
```

分开环境是为了避免 vLLM 与 D3 自定义 Transformers 的依赖冲突。决策环境按固定 D3 模型卡设置 `transformers==5.17.0`；评测固定 `swebench==5.0.2`。vLLM、Torch 及相关 CUDA kernel 的最终构建须在目标设备确认支持这两个 Qwen 架构后固定；当前宽松依赖范围只是安装入口，不是已经验证的完整 lockfile。若稳定版 vLLM 尚不支持指定架构，使用官方要求的构建并记录准确版本/commit，不能静默换模型。

运行前核对 [vLLM 支持模型](https://docs.vllm.ai/en/latest/models/supported_models.html)、[Qwen3.5-9B 模型卡](https://huggingface.co/Qwen/Qwen3.5-9B)、[Qwen3.8-27B 模型卡](https://huggingface.co/Qwen/Qwen3.8-27B) 及 [固定 D3 说明](https://huggingface.co/vllm-sr/d3/blob/cbff3a3cb0b074b6585fe9f747b9f9f5557a0e75/README.md)。D3 会执行固定 revision 中的官方自定义模型代码，并要求原生 loader 完整校验 manifest。

## 2. 配置目标资源并采集设备信息

```bash
cp configs/gpu.example.json configs/gpu.local.json
```

编辑 `configs/gpu.local.json`：

- `runtime.workspace`：结果目录，**相对于配置文件所在目录**；默认 `../work` 即项目内 `work/`。改实验定义或源码后使用新目录，避免混入冻结运行。
- `runtime.hf_cache_dir`：目标 Linux 的模型缓存目录，或 null 使用 HF 默认路径。
- `runtime.gpu_roles.*.visible_devices`：分配给每个角色的 GPU 编号或 UUID。生成器可设 `"0,1"` 配 `tensor_parallel_size: 2`；D3 每次只能指定一张，内部设备 `cuda:0` 指映射后的第一张。
- 生成器 `max_model_len` 必须与 rollout 的 `generator_max_context_tokens` 一致；两个选择器 `question_context_tokens` 必须与公共决策预算一致。
- `runtime.docker` 的内存/CPU/PID 上限；`runtime.evaluation` 的并发与超时。评测命令在 evaluation 环境执行，`python: null` 使用当前环境。
- 两个服务端口默认 8001/8002。当前实现要求同机、受本项目 `serve-generator` 管理的 loopback 服务。
- 仅在目标设备配置完成后，将 `execution_enabled` 改为 `true`。

默认按**顺序加载一个角色**，不要求四个模型同时驻留显存。生成器支持 TP，当前官方 D3 loader 是单设备接口；不能把两张较小显存卡的容量直接相加来部署大 D3。大 D3 仅 BF16 主干权重约 52 GB（十进制），还需要运行空间；80GB 级单卡是可评估的起点，实际能否容纳既定上下文必须实测。本项目不自动量化、不 CPU 卸载。

```bash
.venv-evaluation/bin/d3study --config configs/gpu.local.json validate-config
mkdir -p device-records
.venv-generator/bin/d3study --config configs/gpu.local.json doctor --output device-records/generator.json
.venv-decision/bin/d3study --config configs/gpu.local.json doctor --output device-records/decision.json
.venv-evaluation/bin/d3study --config configs/gpu.local.json doctor --output device-records/evaluation.json
```

`doctor` 只采集 GPU/驱动、Python、依赖版本、Docker、磁盘和配置，不加载模型、不推理。它不证明模型能放入显存，也不代替联调。离线查看启动参数可使用 `serve-generator --generator generator_small --print-command`；占位缓存路径不会被执行。

## 3. 两题联调：逐阶段执行

准备数据与官方实例镜像，在 evaluation 环境执行：

```bash
.venv-evaluation/bin/d3study --config configs/gpu.local.json prepare-data --stage debug
.venv-evaluation/bin/d3study --config configs/gpu.local.json prepare-images --stage debug
```

启动小生成器服务器，保持前台运行（建议在 tmux 或集群作业里）：

```bash
.venv-generator/bin/d3study --config configs/gpu.local.json serve-generator --generator generator_small
```

另一个终端执行生成命令。待 `server.log` 显示服务 ready 后启动；若提前连接失败，只需重新发起客户端命令。

```bash
.venv-generator/bin/d3study --config configs/gpu.local.json generate --stage debug --generator generator_small
```

完成后对服务器按 Ctrl+C 释放显存，再启动大生成器并在另一终端生成：

```bash
.venv-generator/bin/d3study --config configs/gpu.local.json serve-generator --generator generator_large
# 另一终端：
.venv-generator/bin/d3study --config configs/gpu.local.json generate --stage debug --generator generator_large
```

大生成器结束后停止其服务器，再依次运行两个选择器和评测：

```bash
.venv-decision/bin/d3study --config configs/gpu.local.json select --stage debug --selector decision_small
.venv-decision/bin/d3study --config configs/gpu.local.json select --stage debug --selector decision_large
.venv-evaluation/bin/d3study --config configs/gpu.local.json evaluate --stage debug
.venv-evaluation/bin/d3study --config configs/gpu.local.json analyze --stage debug
.venv-evaluation/bin/d3study --config configs/gpu.local.json status --stage debug
```

联调需确认两种模型返回的 native reasoning 非空且与 final 分离、命令真实执行、补丁可提取、D3 真实 Choice 调用成功、官方 harness 产生有效标签、续跑不重采样完成槽位。只跑两题并不能证明统计效果。遇到模型无法加载、没有合法动作、上下文超限或评测错误，应修复并用新 workspace 重做 debug，不能把基础设施失败描述为协作有效性。

## 4. 冻结后跑 50 题

联调完成后固定依赖构建、硬件映射、提示、上下文/输出预算、超时、采样参数、精度、失败策略和总资源预算。用各环境的 `pip freeze` 留存依赖；先检查文件没有私有带凭据 URL，再分享。`doctor` 只输出选定包的版本和凭据是否存在，不输出凭据值。

当前默认生成上下文 32768、每回复最多 16384 输出 token（包含 thinking）、每轨迹最多 131072 输出 token/100 步/3600 秒；选择器公共输入预算 32768 token。这是研究预算，不是模型能力上限。若需要更长上下文，应在 debug 前调高并测显存；不要在 50 题中途更改。历史仅按完整 assistant/observation 对裁剪，首条任务保留；决策器完整 issue/diff 超预算则记 context failure，不截断补丁。

先执行 `prepare-data --stage pilot`、`prepare-images --stage pilot`，再按上节的相同顺序依次执行两次 `generate`、两次 `select`、`evaluate`、`analyze`，将 `--stage debug` 改为 `--stage pilot`。`serve-generator` 不区分阶段。各阶段命令均显式发起，不会在 debug 之后自动启动 pilot。

## 结果、重试与长任务

结果位于 `work/runs/{debug,pilot}/`：

- `frozen.json` 固定配置、程序/运行提示 hash、数据/镜像 manifest hash。
- `pools/` 保存每个槽位全部 attempt 的原始响应、reasoning、命令、观察、补丁和时间/token 账目。
- `decisions/` 保存呈现映射、选择/失败类别、原始 Choice 结果和耗时；候选生成器身份不进入模型输入。
- `evaluation/` 保存决策封存、每次 harness 尝试及标签。重发 `evaluate` 只重试缺失或无效报告，保留已有有效标签。
- `analysis/report.md` / `summary.json` 输出 Single/随机/双选择器/Oracle、按题配对 bootstrap、有效决策分母、失败类别、全部生成尝试资源账目。

同命令续跑会跳过完成槽位，先提升已经写完但尚未发布的 attempt。仅基础设施失败或中断可消耗最多两次尝试，格式失败、重复补丁、截断和错误解题不补采样。中断导致结果未知的模型请求保留输入计数和输出上界，不当作零成本。请求耗时不等于 GPU 活跃秒；后者应从调度器或 profiler 采集。基础设施错误在主系统指标中按失败，同时单独报告；有缺失评测标签时 Oracle 只是已观察覆盖的下界。

服务器与客户端都放入 tmux/Slurm/systemd 等持久作业。SSH 断开、调度器超时或重启不会自动重启进程；恢复后先启动对应服务再重复未完成阶段。`status` 以心跳、主机、boot ID 和 PID 验证是否仍运行，历史 `running` 不会被直接当作存活。服务器日志位于 `work/services/{role}/server.log`。强制终止时由作业管理器结束整个进程组；不要只终止父进程而遗留 GPU server。闲置解题容器有 TTL，正常/可捕获中断会清理。

生成命令会核对本机受管服务的固定快照启动记录、存活 PID/boot/心跳及 HTTP model 身份。此为受信任本地操作员下的来源关联，不是远程密码学证明。不能自行将不相关服务伪装成同名 alias。

## 本地验证与研究文档

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

模拟测试覆盖 thinking/final 边界、固定版本与 CUDA 拒绝回退、原生 D3 上下文预检、共享池、封存、恢复、评测重试和成本记账；不访问模型或 Docker。[PLAN.md](PLAN.md) 为研究设计，[PROMPTS.md](PROMPTS.md) 解释协议；实际执行提示以 `src/d3study/prompts/` 为准。CLI 只使用带 `runtime` 的 `configs/gpu.local.json`，由唯一模板 `configs/gpu.example.json` 复制。
