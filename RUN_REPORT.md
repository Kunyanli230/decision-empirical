# 联调记录（截至 2026-10-27 17:26 +08:00）

这份记录抄自当时还在的作业和目录。若 `/scratch/26044022g` 被清掉，用这里的事实恢复，不要把未完成的第二次运行说成已经评测完。

## 部署

- 集群：PolyU HPC，`hpclogin.comp.polyu.edu.hk`，用户 `26044022g`。计算分区 `universal`，一次只用一张 `gpu:7g.80gb`。
- 本机 RTX 4060（8 GB）装不下 D3，两个模型都在 80 GB MIG 上。
- 小生成器 G_S：Ollama `qwen3.5:2b`（Q8_0），`think: true`。二进制 `/scratch/26044022g/ollama/bin/ollama`，模型目录 `/scratch/26044022g/ollama/models`。GGUF 里的 `qwen35.context_length` 是 262144。
- 大决策模型 D_L：原生 transformers 加载 `vllm-sr/d3@cbff3a3cb0b074b6585fe9f747b9f9f5557a0e75`。权重在 `/scratch/26044022g/hf/models--vllm-sr--d3/snapshots/cbff3a3cb0b074b6585fe9f747b9f9f5557a0e75`。`text_config.max_position_embeddings` 是 262144。单独加载成功过：设备 `NVIDIA A800-SXM4-80GB MIG 7g.80gb`，`native_reported_model` 为 `d3`。
- 本轮不部署 G_L（Qwen3.8-27B / vLLM）和 D_S（d3-flash）。
- CUDA 13 的用户态兼容库：`/scratch/26044022g/cuda-compat/usr/local/cuda-13.0/compat`，必须在进程启动前放进 `LD_LIBRARY_PATH`。驱动是 535。Ollama 仍会打印 “driver too old / required 550”，随后走 `driver=13.0` 那条路径，GPU 推理是通的。
- 计算节点没有可用的 Docker（用户不在 `docker` 组）。评测镜像用 Pyxis 导出后再用 Enroot 打开。
- 镜像：`docker.io/swebench/sweb.eval.x86_64.pydicom_1776_pydicom-955:latest`。squashfs：`/scratch/26044022g/enroot/pydicom-955.sqsh`（约 2.5 GB），链接为 `/scratch/26044022g/enroot/data/pydicom__pydicom-955.sqsh`。文件 sha256：`b5e6e4da4d99eecb86e3e463edece4e7e86d890f2948dadcb890cb9e536abbb8`。Enroot 自己的 `import` 读不了这个单 manifest 镜像；不要再走那条导入。
- 容器里 conda 在 `/opt/miniconda3`，环境名是 `testbed`。`/tmp` 不保留，`/testbed` 和 `/root` 会保留。
- 评测用的 Python：`/scratch/26044022g/venvs/evaluation/bin/python`（conda-forge 3.11，不是登录节点专有的 python311）。决策环境：`/scratch/26044022g/venvs/decision`。

## 题目

- `pydicom__pydicom-955`
- 仓库 `pydicom/pydicom`，版本 1.3
- `base_commit`：`fdd4fc8098920b1cda6127bdc05ff1e542b519fb`
- `FAIL_TO_PASS`：`pydicom/tests/test_filereader.py::TestReader::test_lut_descriptor`
- `PASS_TO_PASS`：354 条
- 数据集行没有 `eval_script`。评分脚本是自行拼的 pytest，再交给 `swebench` 的 `parse_log_pydicom` 和 `get_eval_report`。官方 Docker harness 没有跑。
- 镜像在 Enroot 里能 `git reset --hard` 到上述 base commit。当时 HEAD 信息是 `Fix pickling modified datasets (#952)`。

## 第一次跑完的结果（空补丁）

目录：`/scratch/26044022g/work/runs/debug`。配置当时是每次回复最多 16384 token（含 thinking），上下文 32768。三条轨迹的补丁都是空的，sha256 都是空字符串的 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`。

| 候选 | 结束原因 | 已执行动作 | 格式错误 | 输出 token |
|---|---|---:|---:|---:|
| 0 | `generation_truncated_before_action` | 1 | 0 | 16816 |
| 1 | `repeated_format_error` | 0 | 3 | 2573 |
| 2 | `generation_truncated_before_action` | 0 | 0 | 16384 |

候选 0 的第一条动作是合法 bash，探测 LUT。下一次生成把 16384 token 都用在推理里，`content` 为空。推理结尾在重复同一段字节序分析。候选 1 三次都只输出了没有围栏的 `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`。候选 2 第一次生成就在推理里用满 16384，没有执行命令。

D_L 加载成功后发现三份 diff 相同，状态 `identical_pool`，`selected_index` 为 0，没有做 Choice。评测标签三份都是 `resolved: false`，`source: empty_or_protocol_invalid`。因为补丁为空，没有进 pytest。`evaluation_errors` 为 0。分析摘要在 `/scratch/26044022g/work/runs/debug/analysis/summary.json`。生成侧 6 次请求，输入 4208 token，含推理的输出 35773 token，请求耗时约 933 秒。

这不是轨迹去重算法。全文轨迹级 Best-of-N 没有专门的去重；diff 不足两个 unique 候选时不叫决策模型，记为没得比。论文里的去重在步级 action：token 集合 Jaccard ≥ 0.8 丢掉，同一状态回放后非空 observation 相同再删，少于两个 unique 才不做验证。本地代码已按这个规则改了选择入口，但下面这次正在跑的进程是按启动时冻结的旧代码跑的，选择阶段不会自动用上新规则。

## 正在跑的第二次（不要把它写成已完成）

- Slurm 作业 `21000`，名 `d3-one-q`，状态 RUNNING，节点 `hpcnode2`。17:26 左右已跑约 30 分钟。时限 72:00:00。`srun` 的父进程是 1，关掉本地电脑不会停它。
- 工作区：`/scratch/26044022g/work-r2`。日志：`/scratch/26044022g/logs/one-question-native.log`。Ollama 日志：`/scratch/26044022g/work-r2/services/generator_small/server.log`。
- 启动：2026-10-27T16:56:14+08:00。Ollama 在 16:56:21 就绪。`OLLAMA_CONTEXT_LENGTH=262144`，`num_ctx=262144`。一次回复可以使用提示之外剩下的上下文，不再另加 16384/24576 封顶。单次请求超时 14400 秒，单条轨迹墙钟 86400 秒。
- 阶段：`generate`，候选 0，第 2 步（step 1）仍在生成。心跳正常。已有 1 次模型回复和 1 次执行；events 约 9152 字节。Ollama 这一步大约已生成 56916 token，约 35 token/s。输出上限约 261411，这一步还可以继续很久。
- 第 1 步已执行，返回码 1。命令在 `/testbed` 里用一段显式 VR=SS 的 LUT 字节调用 `dcmread`。断言期望 `[62720, -2048, 16]`，实际是 `[-2816, -2048, 16]`。这是缺陷复现，不是基础设施失败。当时还没有最终补丁，也没有选择，也没有第二次评测。
- 候选 1 和候选 2 还没开始。

## 还没发生的事

- 第二次运行还没有 `result.json`、决策文件或评测标签。
- 还没有在步级对多个 action 做 Jaccard 或相同 observation 删除。正在跑的仍是每步一个动作、一共三条完整轨迹。
- 官方 `swebench.harness.run_evaluation`（Docker）没有跑成。
