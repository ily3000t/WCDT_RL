# WCDT_ACCVP / SAFE_RL

本仓库研究 SUMO `highway_merge` 场景中的安全强化学习。当前正式实验主线是
**ACCVP VNext**：用 action-conditioned Candidate Table 估计候选动作的合流可行性，
由独立 Risk/Safety Shield 保持局部安全权威，再通过成对、多训练副本的闭环评估检验收益。

旧 WcDT、schema-v2 ACCVP、Reward-v2/v3 开发实验仍可用于复现和故障诊断，但不能直接
支持 VNext calibration、operating point、confirmatory evaluation 或部署结论。

## 当前方法边界

VNext 将职责明确拆开：

```text
Risk Module / Safety Shield  -> 局部安全否决与 fallback
ACCVP Candidate Table       -> 每个候选动作的任务可行性与进入时间
PPO                          -> 在固定 observation/reward 契约下学习动作策略
Stage5 + final holdout       -> 独立确认与一次性最终评估
```

ACCVP safety head 只用于诊断，不能替代 Risk Module 的硬安全门控。完整方法也不等于
“Candidate Table + Reward-v2”：正式 Reward 因子角色为：

| 方法 | 角色 |
| --- | --- |
| WcDT + Reward-v2 | action-independent forecast baseline |
| Candidate Table + Reward-v2 | Candidate Table 单因素消融 |
| Candidate Table + Reward-v2 + commitment | commitment 因素对照 |
| Candidate Table + Reward-v3.1 | persistence reward 因素对照 |
| Candidate Table + Reward-v3.1 + risk-gated commitment | 最终完整方法候选 |

历史实验只支持将 Reward-v3.1 与 commitment **预注册为候选**；它们仍须在 schema3、
无泄漏 split、五个 optimizer seeds 和独立 confirmatory cohorts 下重新验证。

## 正式 VNext 入口

正式状态由
[`safe_rl/config/active/accvp_vnext/workflow.yaml`](safe_rl/config/active/accvp_vnext/workflow.yaml)
和产物报告共同决定。先查看状态，不会启动任务：

```powershell
python -m safe_rl.pipeline.run_accvp_vnext_pipeline
```

只执行当前一个阶段：

```powershell
python -m safe_rl.pipeline.run_accvp_vnext_pipeline --execute-next
```

连续执行到指定阶段：

```powershell
python -m safe_rl.pipeline.run_accvp_vnext_pipeline --run-until pilot_validation
```

`--run-until` 不是无条件一键跑完。它会在以下任一情况立即停止：

- 子进程失败；
- 阶段返回后 artifact gate 仍关闭；
- workflow 将下一阶段标为 blocked；
- 到达指定阶段；
- sealed final holdout 未得到显式授权。

最终 holdout 只有在全部上游门槛通过并冻结后才能显式打开：

```powershell
python -m safe_rl.pipeline.run_accvp_vnext_pipeline `
  --execute-next `
  --allow-final-holdout
```

当前 workflow 会在 `candidate_ppo_replicates` 前 fail closed，直到 factorial PPO/Stage5
协调器能够把 Reward-v2 固定为消融、把 Reward-v3.1 + commitment 固定为最终候选。
这避免现有单一 Reward-v2 命令被误当作完整方法。通用
`safe_rl.pipeline.run_full_pipeline` 仍可用于旧版/一般比较实验，但不是 VNext 正式证据入口。

## 正式阶段顺序

```text
pilot collection / merge
  -> seed 2/5 oracle-only regression
  -> pilot validation
  -> formal schema3 collection / merge
  -> ACCVP ensemble training, calibration, operating point
  -> scorer runtime preflight
  -> factorial PPO, five optimizer seeds per method
  -> policy runtime benchmark (worst replicate gate)
  -> paired Stage5 reports and hierarchical aggregation
  -> sealed one-shot final holdout
```

每次推进前建议重新运行状态命令，确认 `next_phase`、`next_command`、`blocked` 和对应
artifact。不要因为 report 为 `fail` 就删除数据、放宽阈值或跳过阶段；应读取报告的
`conditions`/`gate` 定位唯一关闭项。

## 不可绕过的数据契约

- 正式反事实数据必须是 schema3；旧 schema2 数据只能用于 diagnostic。
- actor rows 只能由 `selected_indices` 生成，并由 mapping hash 约束。
- split component 联合绑定 model-input fingerprint 与 scenario/traffic/episode seed。
- ensemble bootstrap 以 fingerprint component 为单位，重复样本组总权重为 1。
- target-lane entry time 只监督 `observed_success`；失败或 censored 分支不得伪造进入时间。
- seed 2/5 是 `oracle_regression` cohort，必须 `oracle_only=true` 且排除全部模型 split。
- calibration、operating-point selection 和 test/holdout cohorts 必须隔离。
- operating-point 的 0.95 model-availability gate 以 Risk-eligible decision 为分母；完整运行
  覆盖率与 Risk-ineligible 比例必须另行报告，后者继续 fail closed。
- final holdout 不能生成阈值、改配置或触发重训。

完整契约与故障恢复说明见
[`safe_rl/accvp/README.md`](safe_rl/accvp/README.md)。

## 配置与产物

配置状态以 [`safe_rl/config/registry.yaml`](safe_rl/config/registry.yaml) 为准：

```text
safe_rl/config/
  active/accvp_vnext/  # canonical VNext configs and workflow
  baselines/           # maintained comparison arms
  examples/            # templates, not frozen experiments
  archive/             # diagnostic_only historical configs
  local/               # gitignored machine-specific copies
```

不要根据文件名、能否被 YAML loader 读取或是否位于仓库中推断正式地位。生成的 PPO
replicate 和 Stage5 配置写入 `safe_rl_output/runs/.../generated_configs`，不回写 canonical
配置目录。既有 run 目录包含路径和 hash lineage，不应为了配合新目录结构而移动。

VNext 产物使用 generation-aware 名称：

```text
accvp_vnext_schema3_predictor.pt
accvp_vnext_schema3_calibration.json
accvp_vnext_schema3_candidate_manifest.json
```

配置分类细节见 [`safe_rl/config/README.md`](safe_rl/config/README.md)。

## 环境

推荐从仓库环境创建 Python 环境，并使用同一个解释器执行所有阶段：

```powershell
conda env create -f environment.yml
conda activate WcDT
$env:SUMO_HOME = "<SUMO installation>"
sumo --version
python -m pytest -q
```

Windows 正式 CUDA 训练应在新的 Python 进程启动前设置：

```powershell
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
```

不要在一次流程中混用不同 Conda 环境、SUMO binary 或 TraCI 来源。当前 active 配置仍
包含项目本机既有 checkpoint 路径；换机器时应先完成 lineage 审计，而不是静默替换模型。

## 代码结构

```text
safe_rl/accvp/                 # schema3、dataset、split、模型、训练与 artifact lifecycle
safe_rl/stage1_counterfactual/ # snapshot/branch collection
safe_rl/risk/                  # Risk Module 与 Candidate Risk backend
safe_rl/shield/                # Safety Shield 与 bounded-stale 行为
safe_rl/rl/                    # PPO 训练和 observation/reward integration
safe_rl/analysis/              # paired/hierarchical statistics
safe_rl/pipeline/              # 可执行阶段入口与 VNext coordinator
safe_rl/config/                # registry、active、baseline、example、archive
scenarios/highway_merge/       # SUMO network、routes、scenario config
tests/                         # 数据、模型、runtime、统计和协调器回归测试
```

## 测试

代码修改后至少执行：

```powershell
python -m compileall -q safe_rl tests
python -m pytest tests/test_accvp_vnext_pipeline.py tests/test_accvp.py -q
python -m pytest -q
```

测试通过只证明代码级契约；它不能替代真实 SUMO runtime benchmark、五副本 Stage5 或
sealed holdout。synthetic fault audit 也只验证 bounded-stale 状态机，不构成 hard
real-time deployment claim。

## 历史结果与原始 WcDT

旧 run 可以用于方法演进、failure-case regression 和论文结果溯源，但必须标为
`diagnostic_only`。特别是受 schema2 actor-row 不确定性、fingerprint split 泄漏、test
参与阈值选择或单 member uncertainty 影响的结果，不能升级为 VNext 正式结论。

仓库保留原始 WcDT 网络和数据生成代码作为上游研究基础。使用上游工作或本仓库实验时，
请分别遵循相应许可证与引用要求；VNext 报告必须同时保存配置、checkpoint、数据与报告 hash。
