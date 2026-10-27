# 新设备需要确认的信息

本清单用于迁移 GPU 实验；不根据当前 Windows 电脑的配置选型。代码已实现，**尚未做真实 GPU 联调**。先收集下列信息，再决定 GPU 映射、TP、上下文和并发；详细执行顺序见 [README](README.md)。

## 可直接填写的设备记录

```text
设备/集群名称：
Linux 发行版 / 内核 / glibc / x86_64：
Python 版本：
GPU 厂商、型号、张数：
每张 GPU 总显存 / 空闲显存 / 分配的编号或 UUID：
MIG 配置 / 是否共享 / 其他常驻进程：
驱动版本 / Torch CUDA 构建 / BF16 支持：
多卡互联（NVLink / PCIe）及 TP 可用卡数：
CPU 核数 / 内存：
模型缓存目录 / 空闲空间 / 配额：
Docker Root Dir / 空闲空间 / 配额：
项目与结果目录 / 空闲空间 / 是否持久存储：
Docker 版本 / daemon 访问权限 / x86_64 镜像支持：
外网或镜像源 / HF 访问 / Docker Hub 拉取限制：
任务管理器 / 单作业 walltime / GPU 时数或费用上限：
SSH 断开、抢占、重启后的恢复方式：
生成环境 vLLM / Torch / Torchvision / CUDA kernels 版本：
决策环境 Transformers / Torch / Torchvision / HF Hub 版本：
评测环境 swebench / datasets / Docker 版本：
G_S GPU IDs / TP / 上下文：
G_L GPU IDs / TP / 上下文：
D_S 单 GPU ID / 公共决策上下文：
D_L 单 GPU ID / 公共决策上下文：
Docker CPU、内存上限 / 评测并发：
最终总预算 / 停止条件 / 配置与依赖冻结位置：
```

不要填写任何 API key、HF token 或含凭据的代理 URL。本研究使用本地 D3 权重。

## GPU 与显存

1. 当前支持路径是 Linux x86_64、NVIDIA CUDA 和 BF16。GPU 型号、驱动、Torch/vLLM CUDA 构建必须匹配；AMD/ROCm 不属于本轮已准备的验收路径，不能仅看到 `torch.cuda` 接口就视为兼容。
2. 原版大 D3 是约 26.09B 参数的单设备原生 loader，BF16 主干权重本身约 **52 GB 十进制**，还要加激活、缓存、FP32 读出和框架开销。80GB 级单卡可作为待测候选，不能保证给定上下文一定放得下。小 D3 约 8.39B，对应主干权重约 17 GB，仍需额外空间。
3. Qwen 9B/27B 的 BF16 权重分别约 18/54 GB，实际总量依检查点而定；vLLM 还需要 KV cache 等空间。生成器可以 TP；**当前 D3 没有实现 TP 或跨卡切分**。两张 48GB 卡并不自动满足大 D3 的单设备要求。
4. 本实现按角色顺序加载。生成结束需停止 vLLM server，再加载选择器。确认显存是否被其他作业占用，GPU 编号在调度环境是否被重映射，MIG 实例是否有足够容量。
5. thinking 增加输出长度、延迟和 KV 占用。默认上下文 32K 是研究预算；调大前先确认显存。不能通过静默关闭 thinking、量化或 CPU 卸载来绕过本轮设置。

## 软件与依赖

- 使用三个独立环境：generator、decision、evaluation/data。建议 Python 3.11；最终版本和 ABI 兼容性在目标设备确认。
- 确认 vLLM 构建支持 **两个指定 Qwen 架构、qwen3 reasoning parser、/tokenize、chat_template_kwargs、reasoning_effort**。实际回复需要分离 reasoning/final；仅 HTTP 200 不算验收。
- 决策环境固定 Transformers 5.17.0，加载固定 D3 自定义代码；需允许 `trust_remote_code=True`，原生 manifest 完整验证、CUDA BF16 加载与 Choice 推理均需通过。
- 评测固定 SWE-bench 5.0.2；Docker daemon 对当前账号可用，能够拉取官方 `swebench/sweb.eval.x86_64.*` 镜像。容器运行项目命令不需要 GPU，GPU 由宿主推理进程使用。
- 保留每个环境的版本记录及冻结依赖。不要假设本机模拟测试代表 CUDA kernel、驱动或模型架构已验证。

## 存储、网络与进程

- 分别确认 HF cache、Docker Root Dir、workspace 的实际文件系统与空闲配额。模型四套 BF16 权重合计只是约 140 GB 量级的权重估计，下载缓存、镜像层、日志与重复版本另计；不能用这个数当全部存储预算。先测两题镜像实际占用，再给 50 题预留空间。
- 确认四个固定 HF snapshot 能下载、自定义代码文件齐全，HF/Docker Hub 的访问、认证、代理与 CA 正常。认证只通过环境变量或本地凭据管理。
- 模型服务默认监听本机 127.0.0.1:8001/8002；确认端口无占用。当前 runner 要求同机 `serve-generator` 启动记录，尚不支持直接接入任意远程模型服务。
- 确认任务管理器、队列限时、抢占策略、日志和持久磁盘。将 server/client 都放进持久作业；退出时清理整个进程组。目标联调验证一次断线恢复和一次中断续跑。
- 原生 D3 的 forward 在本进程内执行，没有独立 HTTP 请求超时。需使用调度器 walltime/作业监控处理进程挂起，并保存已完成决策；不要把生成器 HTTP timeout 当作 D3 的超时机制。

## 实验冻结与两题验收

- 确认原版 d3 与 d3-flash 固定 SHA；不用公共 API 的 `d3` alias 替代原版。生成母模型的精确训练初始化 commit 未公开，记录其来源声明与我们选择的公开 snapshot，不能补写成已证实相同初始化权重。
- 检查数据 manifest 的 revision、两道 debug ID、50 道 pilot ID、此前 10 题排除表、Docker image ID/digest。debug 排除整个 Verified 集合。
- 检查两个生成器均 thinking ON，reasoning/final 分开；完整轨迹 N=3、种子、预算、上下文、动作协议、决策输入、失败策略一致且冻结。
- 真实通过：两种生成器各产生 3 个候选槽位、两种 D3 都调用 Choice、同一池/同一顺序、只执行 final、封存后官方评测、分析报告和未知成本计数正常。无效动作、截断或空补丁会保留为失败，不为了得到好结果重采样。
- 确认 `status` 心跳有效、重发阶段不会重采样完成轨迹、失败评测可重试。只有两题链路验收后，才显式开始 50 题；50 题是探索性结果，不能据此声称大小的纯因果效应。

## 收集信息的只读命令

在目标设备运行，下列命令本身不加载模型或执行实验：

```bash
uname -a
ldd --version
python3.11 --version
nvidia-smi -L
nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.free,driver_version --format=csv
nvidia-smi topo -m
free -h
df -h
docker version
docker info --format '{{.Architecture}} {{.DockerRootDir}}'
docker system df

.venv-generator/bin/d3study --config configs/gpu.local.json doctor --output device-records/generator.json
.venv-decision/bin/d3study --config configs/gpu.local.json doctor --output device-records/decision.json
.venv-evaluation/bin/d3study --config configs/gpu.local.json doctor --output device-records/evaluation.json
```

先创建 `device-records/` 并确保已复制本地配置。`doctor` 自动创建输出父目录，只记录凭据是否存在。它收集的是环境信息，无法自动确定全部作业配额或验证模型显存需求。
