# 轨迹级 Best-of-3 的提示与输入协议

本项目只使用固定 Qwen 生成母模型与两个固定 D3 选择器。debug 和 pilot 每个生成器每题均生成三条独立完整轨迹；两个选择器只从同一批最终补丁中各选一个。实际提示以运行器打包的两个文本文件为唯一来源，配置、源码及提示 hash 一同冻结。

## 生成器

共用 [actor.txt](src/d3study/prompts/actor.txt)。将 problem_statement 填入任务消息，此后按各模型官方 chat template 追加本轨迹的最终动作和已观察反馈。每次回复的 final content 必须恰好包含一个非空 bash 围栏，外部只允许空白；原生 reasoning 单独保存，不提取其中命令执行。

- G_S：Qwen3.5-9B，enable_thinking=true；官方模板排除历史 thinking，保留最终动作与观察。
- G_L：Qwen3.8-27B，enable_thinking=true、preserve_thinking=true、reasoning_effort=xhigh；保留原生历史推理字段。

候选只见自己的工具交互，不见其他候选、选择器意见、官方评测标签或 gold 字段。三条候选使用相同任务、提示、采样参数和初始提交，随机种子独立。禁止修改或新增测试/评测文件；临时复现放在 /tmp，不能提交 git commit 或获取外部答案。

仅接受以下精确完成脚本，并核对执行成功及输出首行标记：

```bash
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
```

运行器统一提取相对 base_commit 的最终 diff。格式失败只接收固定反馈，不能由其他模型代写动作。达到输出上限但没有完整 final 时记为截断；reasoning 与 final 的全部输出 token 均计入预算。

## 选择器

D_S 与 D_L 共用 [selector.txt](src/d3study/prompts/selector.txt)，通过各自固定权重的原生 `system_one` Choice 调用，不生成补丁或自由文本解释。

原生输入结构为：

```json
{
  "state": {
    "issue": "<完整问题>",
    "repository": "<仓库身份>",
    "evidence_policy": "Only recorded command results are observations. Official evaluation labels are unavailable."
  },
  "questions": {
    "choice": {
      "type": "choice",
      "instructions": "<selector.txt 的完整内容>",
      "criteria": {
        "option_1": {"diff": "<完整补丁>", "changed_paths": [], "protocol_status": "valid", "termination_reason": "submitted", "observed_checks": []},
        "option_2": {"diff": "<完整补丁>", "changed_paths": [], "protocol_status": "valid", "termination_reason": "submitted", "observed_checks": []},
        "option_3": {"diff": "<完整补丁>", "changed_paths": [], "protocol_status": "valid", "termination_reason": "submitted", "observed_checks": []}
      }
    }
  }
}
```

占位文本不是实际请求。补丁、状态和观察必须由运行器填入真实记录。observed_checks 只取所接受 attempt 最后五次已执行动作，按固定字符预算截取命令与输出并标注截断；不使用 LLM 摘要。完整 issue/diff 不截断，超过公共 token 预算即记录 context failure。

三个候选按预先固定随机数显示为 option_1..3，两个选择器看到完全相同的顺序。原始候选序号、seed、生成器名称、参数规模和官方标签不进入请求。三个槽位均保留，包括重复、空补丁和协议无效候选。

保存完整原生返回，从 `answers.choice` 校验 choice、probabilities、confidence 及 usage；返回 choice 必须映射回已有槽位。confidence 不是通过官方测试的已校准概率。候选全相同则按预定规则使用槽位 0 并记录 identical_pool，不伪造一次模型调用。

## 封存与评测

全部四个生成器×选择器组合的决策封存后，再使用官方 SWE-bench harness 评测所有候选。选择结果按补丁字节和 SHA-256 关联，不重新生成、合并或修补。Single-first、Random@3、Oracle@3 都是同一候选池的分析对照，不增加生成或选择调用。

日志保存完整请求/响应、映射、模型身份、耗时、token 用量和失败类别；凭据不写入产物。真实 GPU thinking 与 Choice 行为仍需目标设备联调验收。
