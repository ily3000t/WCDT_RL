# WCDT_ACCVP / SAFE_RL

本仓库研究 SUMO `highway_merge` 场景中的安全强化学习。当前正式实验主线是
**ACCVP VNext Selector-v3**：用 action-conditioned Candidate Table 估计候选动作的合流可行性，
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

这里的正式 factorial 是 **Candidate Table 固定开启条件下的 Reward × commitment 2×2**，
再加一个 WcDT Reward-v2 公共 baseline。它可以估计 persistence、commitment 及二者交互，
并在 Reward-v2/no-commitment 条件下比较 Candidate Table 与 WcDT；它不是完整的
Candidate Table × Reward × commitment 三因素设计。

## 正式 Selector-v3 入口

正式状态由
[`safe_rl/config/active/accvp_vnext_selector3/workflow.yaml`](safe_rl/config/active/accvp_vnext_selector3/workflow.yaml)
和产物报告共同决定。协调器有三种运行方式。

### 1. 只查看状态

不会启动任务，也不会修改 artifact：

```powershell
python -m safe_rl.pipeline.run_accvp_vnext_pipeline
```

重点读取输出中的 `next_phase`、`next_command`、`blocked`、`blocked_reason` 和各阶段
`complete/artifact`。

### 2. 只执行下一个阶段

```powershell
python -m safe_rl.pipeline.run_accvp_vnext_pipeline --execute-next
```

命令最多执行当前 `next_phase` 一次。子进程成功返回后仍会由下一次状态查询验证 artifact gate。

### 3. 连续执行到指定阶段

`--run-until <phase>` 会从当前第一个未完成阶段开始，连续执行并包含目标阶段。例如：

```powershell
# 完成 schema3 pilot、oracle 与 pilot gate
python -m safe_rl.pipeline.run_accvp_vnext_pipeline --run-until pilot_validation

# 从当前状态连续执行到 formal dataset merge 完成
python -m safe_rl.pipeline.run_accvp_vnext_pipeline --run-until formal_merge

# 包含 ACCVP ensemble training/calibration/operating-point
python -m safe_rl.pipeline.run_accvp_vnext_pipeline --run-until accvp_training

# 训练完成后，继续执行真实 SUMO scorer runtime preflight
python -m safe_rl.pipeline.run_accvp_vnext_pipeline --run-until scorer_runtime_preflight

# 训练四个 Candidate factorial 方法，每个方法五个 optimizer seeds（共20个）
python -m safe_rl.pipeline.run_accvp_vnext_pipeline --run-until candidate_ppo_replicates

# 完成 WcDT baseline 与四个 Candidate 方法的 policy runtime gates
python -m safe_rl.pipeline.run_accvp_vnext_pipeline --run-until policy_runtime_replicates

# 生成、执行并聚合六个预注册 Stage5 比较
python -m safe_rl.pipeline.run_accvp_vnext_pipeline --run-until stage5_replicates_and_aggregate
```

目标阶段若已经完成，协调器只输出 `reached target`，不会重跑或覆盖已有 artifact。目标名必须
来自下面的 phase 表；拼写错误会列出全部合法值。

`--run-until` 不是忽略报告的一键跑完。它会在以下任一情况立即停止：

- 子进程失败；
- 阶段返回后 artifact gate 仍关闭；
- workflow 将下一阶段标为 blocked；
- 到达指定阶段；
- sealed final holdout 未得到显式授权。

### 阶段与可用的 `--run-until` 目标

| 顺序 | phase | 执行内容与完成标志 |
| ---: | --- | --- |
| 1 | `selector_contract_audit` | 先审计旧 5,000 root 的全量 current/history 输入，再在 1002/1004 × 50021/50027 四个 diagnostic replay 上并行 shadow selector-v2/v3；严格冻结容量 6 或 8。|
| 2 | `pilot_collection` | 使用全新 Selector-v3 run ID 采集 schema3 pilot immutable shards。|
| 3 | `pilot_merge` | 合并 pilot shards，生成 pilot dataset manifest。|
| 4 | `oracle_collection` | 仅采集 seed 2/5、`oracle_only=true` 的历史 repairability regression。|
| 5 | `oracle_merge` | 合并 oracle-only shards；该 dataset 永不进入模型 split。|
| 6 | `oracle_regression` | 生成 oracle report；`oracle_state` 必须为 `go`。|
| 7 | `pilot_validation` | 审计 schema、actor mapping、task/Risk coverage、branch throughput 和 oracle exclusion；`pilot_state=pass`。|
| 8 | `formal_collection` | 按已通过 pilot 的冻结 selector/capacity 契约采集 5,000-root formal shards。|
| 9 | `formal_merge` | 合并 formal shards，生成 dataset provenance；不会训练模型。|
| 10 | `formal_validation` | 在训练前强制检查 rejected/overflow/task coverage/Risk coverage、branch success 与无泄漏 split；`formal_state=pass`。|
| 11 | `accvp_training` | 训练三成员 ensemble，在独立 split 上 calibration 与 operating-point selection，生成 sealed candidate bundle。|
| 12 | `scorer_runtime_preflight` | 使用 rule policy 访问真实 SUMO state，只测 ACCVP/Risk Candidate Table；正式 runtime 使用未观察过的 55001–55060，且旧 implementation 的失败报告只归档、不复用 episodes。|
| 13 | `candidate_ppo_replicates` | 冻结 factorial plan，顺序训练四个 Candidate 方法 × 五个 optimizer seeds，共 20 个唯一 checkpoints。|
| 14 | `baseline_ppo_replicates` | 复用已有五副本 WcDT manifest；不完整或有冲突时 fail closed，不覆盖现有目录。|
| 15 | `baseline_lineage_audit` | 独立核验五个 WcDT checkpoint/config/report hash、Reward/observation/预算和 optimizer seed；只有 `status=reusable` 才放行。|
| 16 | `policy_runtime_replicates` | 对四个 Candidate 方法的 20 个 PPO checkpoints 分别执行完整 policy runtime benchmark，总 gate 取所有方法最差结果。|
| 17 | `stage5_generate` | 生成六个冻结比较及每个 training seed 的 paired 配置。|
| 18 | `stage5_replicates_and_aggregate` | 可恢复地运行六组 Stage5 并进行 replicated aggregation。|
| 19 | `one_shot_final_holdout` | 只绑定最终方法与主要 comparison；在全部冻结后显式打开一次。|

当 `accvp_training` 已完成且状态显示 `next_phase=scorer_runtime_preflight` 时，下一步可以二选一：

```powershell
# 只运行 scorer preflight
python -m safe_rl.pipeline.run_accvp_vnext_pipeline --execute-next

# 等价地，明确指定停止目标
python -m safe_rl.pipeline.run_accvp_vnext_pipeline --run-until scorer_runtime_preflight
```

运行完成后再次执行无参数状态命令，确认 `scorer_runtime_preflight.complete=true` 和下一阶段状态。

### Factorial PPO、runtime 与 Stage5 产物

正常情况下使用总 workflow 入口即可。需要单独恢复 Candidate PPO 阶段时，等价的底层入口为：

```powershell
python -m safe_rl.pipeline.stage3_train_ppo_factorial `
  --config safe_rl/config/active/accvp_vnext_selector3/ppo_candidate_table_full.yaml `
  --matrix safe_rl/config/active/accvp_vnext_selector3/ppo_ablation_matrix.yaml `
  --workflow-config safe_rl/config/active/accvp_vnext_selector3/workflow.yaml `
  --optimizer-seeds 1001 1002 1003 1004 1005 `
  --output-root safe_rl_output/runs/accvp_vnext_selector3_factorial
```

它默认安全续跑：完整 seed 会在重新核验 hash 后跳过。若 Stage3 在写入任何文件前退出，
只留下空的 `run/stage3/` 目录树，协调器会明确输出 `empty_run_shell_removed` 并安全重建；
若目录中存在任意文件、符号链接或不完整 checkpoint/report，则仍然 fail closed，要求先人工
审计并归档，不会自动删除或覆盖。关键产物为：

每个副本的 `run.seed` 固定为 simulator-training cohort 起点，`rl.optimizer_seed` 才随
1001--1005 变化。Stable-Baselines3 构造模型时会把模型 seed 同时传给 VecEnv，因此 Stage3
会在模型构造后、第一次 rollout reset 前显式把 VecEnv 恢复为 `run.seed`；preflight 和最终
lineage 分别校验 `ppo_training` 与 `ppo_optimizer_replicates`，禁止两类 seed 混用。

```text
safe_rl_output/runs/accvp_vnext_selector3_factorial/
  factorial_plan.json
  ppo_factorial_manifest.json
  methods/<method_id>/ppo_replicate_manifest.json

safe_rl_output/runs/accvp_vnext_selector3_runtime/
  factorial_runtime_report.json
  methods/<method_id>/replicated_runtime_report.json

safe_rl_output/runs/accvp_vnext_selector3_stage5/
  generated/factorial_request.json
  generated/comparisons/<comparison_id>/replicated_report.json
  episode_cache/<method_id>/optimizer_seed_<seed>/identity.json
  episode_cache/<method_id>/optimizer_seed_<seed>/episodes/seed_<simulator_seed>.json
  factorial_report.json
```

Stage5 预注册六个比较：Candidate Table（Reward-v2）归因、v2 下 commitment、无 commitment
时 persistence、有 commitment 时 persistence、v3.1 下 commitment，以及 WcDT Reward-v2
对最终完整方法。crossed bootstrap 当前不产生可用于跨比较 Holm 家族的合法 p 值，因此总
报告会明确记录 `performed=false`，不会从置信区间反推或伪造 p 值。

为了避免把六个问题都机械地扩展到 300 个 seeds，workflow 冻结两级预算：五个次要机制比较
各使用同一 confirmatory ledger 的前 100 个 seeds；唯一的主要比较
`wcdt_v2_vs_final_method` 使用相同前缀并扩展到 300 个。每个方法、optimizer seed、checkpoint、
Risk checkpoint 和执行契约共同定义一个 cache identity；每个 simulator seed 独立写入一次且
不可覆盖。因而同一 WcDT 或 Candidate 方法出现在多个比较时会复用完全相同的 episode，
主要比较只补跑缺少的 200 个 seeds。该设计把实际 SUMO 负载从原先的 18,000 episodes 降为
4,500 个唯一 episodes；统计比较仍只读取各自预注册的 seed 子集，不会把次要比较结果混入
主要比较。

最终 holdout 只有在全部上游门槛通过并冻结后才能显式打开：

```powershell
python -m safe_rl.pipeline.run_accvp_vnext_pipeline `
  --execute-next `
  --allow-final-holdout
```

factorial 协调器现已将 Reward-v2 固定为消融，并将 Reward-v3.1 + commitment 固定为最终
候选。`candidate_ppo_replicates` 不再人为 blocked，但任何方法、seed、checkpoint、runtime 或
Stage5 lineage 缺失都会关闭 artifact gate。通用 `safe_rl.pipeline.run_full_pipeline` 仍可用于
旧版/一般比较实验，但不是 VNext 正式证据入口。

每次推进前建议重新运行状态命令，确认 `next_phase`、`next_command`、`blocked` 和对应
artifact。不要因为 report 为 `fail` 就删除数据、放宽阈值或跳过阶段；应读取报告的
`conditions`/`gate` 定位唯一关闭项。

## 不可绕过的数据契约

- 正式反事实数据必须是 schema3 + `accvp_240_v3_conflict_selector`；旧 schema2 和 V1 selector-v2 数据只能用于 diagnostic。
- Selector-v3 使用所有合法 ego candidate swept tubes 与 actor 可达管道的并集；当前不在目标车道不能据此降级为非冲突 actor。
- ACCVP split 使用 `task_actor_coverage_complete`；Risk/Shield 的全车状态完整性由独立的 `risk_safety_actor_coverage_complete` 约束。
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
  active/accvp_vnext_selector3/  # canonical Selector-v3 configs/workflow
  active/accvp_vnext/            # V1 diagnostic-only reproduction
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
safe_rl/accvp/contracts/       # schema3、protocol、runtime/artifact contracts
safe_rl/accvp/data/            # dataset、component split、migration audit
safe_rl/accvp/modeling/        # predictor、loss、checkpoint metadata
safe_rl/accvp/training/        # ensemble training、calibration、tuning
safe_rl/accvp/planning/        # candidate plan、selection、ACV-Shield controller
safe_rl/accvp/serving/         # runtime predictor、worker、159D observation
safe_rl/accvp/evaluation/      # oracle、pilot、diagnostics、benchmarks
safe_rl/accvp/verification/    # synthetic fault injection
safe_rl/stage1_counterfactual/ # root/snapshot/branch/shard data generation
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
python -m pytest tests/test_ppo_factorial_training.py `
  tests/test_accvp_runtime_factorial.py `
  tests/test_stage5_factorial.py `
  tests/test_accvp_vnext_pipeline.py `
  tests/test_evaluation_protocol.py -q
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
