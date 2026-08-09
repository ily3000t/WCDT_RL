# ACCVP VNext protocol and execution guide

本文档是 ACCVP VNext 的详细执行契约。日常入口见仓库根 `README.md`；配置状态以
`safe_rl/config/registry.yaml` 为准；机器生成的 artifact/report 才决定阶段 gate 是否打开。

## Code package layout

ACCVP 按职责分包，根目录只保留公共包入口和本文档：

```text
safe_rl/accvp/
  contracts/       # schema、data/runtime protocol、artifact lifecycle
  data/            # dataset loading、component split、legacy migration audit
  modeling/        # predictor architecture、loss、checkpoint metadata
  training/        # trainer、calibration、availability、tuning、reproducibility
  planning/        # candidate plan、selection、controller、Lite planning
  serving/         # CPU runtime predictor、inference worker、159D observation
  evaluation/      # candidate/final diagnostics、oracle、pilot、runtime-result audits
  verification/    # synthetic fault injection；不产生 hard-real-time claim
  __init__.py
  README.md

safe_rl/stage1_counterfactual/
  collector.py      # root trajectory 与 branch job 调度
  branch_worker.py  # 独立 SUMO counterfactual branch
  root_context.py   # root capture/restore
  snapshot_store.py # snapshot 与 row 原子写入
  shards.py         # immutable shard 与 formal merge
```

职责边界：

- `stage1_counterfactual` 只生成和合并反事实数据；
- `data` 只消费合并后的数据并构造无泄漏 split；
- `modeling` 定义网络与 loss，不负责训练流程或在线超时；
- `training` 生成 predictor/calibration/operating-point bundle；
- `planning` 定义动作计划、选择和 ACV-Shield 逻辑；
- `serving` 负责在线评分、Candidate Table observation 与 bounded stale；
- `evaluation` 只读取冻结产物并生成审计/实验报告；
- `verification` 注入 synthetic failure，不能提升部署声明。

`modeling/model.py` 暂时保留 architecture、loss 和 checkpoint metadata 三者共置，因为它们
共享同一个 architecture/loss version。待出现第二种模型实现时再按 `predictor.py`、`losses.py`
和 `checkpoint.py` 拆分，避免本次纯结构迁移同时改变数值逻辑。

旧的 `safe_rl.accvp.<flat_module>` Python 导入路径已经迁移到上述子包；仓库内 pipeline 和测试
已同步更新。CLI 名称、配置路径、run 目录和 generation-aware artifact 文件名均未改变，因此
旧 Selector-v2/Selector-v3 schema3 数据与 checkpoint 不移动、不删除，但在 Selector-v4 协议下标为
`diagnostic_only`；新证据使用独立 run 目录重新采集和训练。

常用导入迁移：

| 旧路径 | 新路径 |
| --- | --- |
| `safe_rl.accvp.schema` | `safe_rl.accvp.contracts.schema` |
| `safe_rl.accvp.artifacts` | `safe_rl.accvp.contracts.artifacts` |
| `safe_rl.accvp.dataset` | `safe_rl.accvp.data.dataset` |
| `safe_rl.accvp.model` | `safe_rl.accvp.modeling.model` |
| `safe_rl.accvp.train` | `safe_rl.accvp.training.trainer` |
| `safe_rl.accvp.controller` | `safe_rl.accvp.planning.controller` |
| `safe_rl.accvp.runtime` | `safe_rl.accvp.serving.predictor` |
| `safe_rl.accvp.observation` | `safe_rl.accvp.serving.observation` |
| `safe_rl.accvp.candidate_table_diagnostics` | `safe_rl.accvp.evaluation.candidate_table` |
| `safe_rl.accvp.fault_injection` | `safe_rl.accvp.verification.fault_injection` |
| `safe_rl.accvp.branch_worker` | `safe_rl.stage1_counterfactual.branch_worker` |
| `safe_rl.accvp.shards` | `safe_rl.stage1_counterfactual.shards` |

## 1. Compatibility boundary

VNext 正式数据和模型必须同时满足：

- `artifact_generation: vnext_schema3`
- `schema_version: 3`
- `data_contract_version: accvp_240_v4_lane_aware_capacity_audit`
- `accvp.actor_relevance.version: merge_conflict_relevance_v4_lane_aware`
- `actor_row_mapping_version: selected_indices_v2`
- `root_observation_fingerprint_version: model_input_fingerprint_v3`
- `entry_time_label_version: conditional_entry_time_v1`
- `loss_version: accvp_loss_v2`

旧 schema2 数据没有足够证据证明 `selected_actor_ids`、branch response rows 与 WcDT token
rows 的唯一映射，因此不能迁移为完整 VNext actor-response supervision。它只允许在明确
标注的 diagnostic/migration audit 中使用。

Selector-v2/V3 数据虽然是 schema3，但 actor-response rows 只覆盖旧 selector 选择的 actor，
不能迁移成 Selector-v4 formal supervision。旧 predictor、Candidate PPO、runtime 和 Stage5
报告只允许用于历史复现、失败诊断和明确标注的 selector state-source audit。

`prediction.actor_relevance` 与 `prediction.wcdt_v3_max_agents` 属于既有 WcDT checkpoint
契约；`accvp.actor_relevance` 与 `accvp.actor_count` 属于 Candidate Table 数据/推理契约。
两者不得通过修改 checkpoint hash 校验来混用。Selector-v4 审计只冻结 ACCVP 容量，
不会改变旧 WcDT 的 selector-v2、6-actor lineage。

## 2. Data semantics

### Actor mapping

`selected_indices` 是 actor row 的唯一权威来源。root tensor、selected actor IDs、branch
response rows 和 token rows 必须共享 mapping hash。缺失或不一致时应立即拒绝样本。

### Entry-time labels

目标车道进入时间是 conditional label：

- `observed_success`：允许 `target_lane_entry_time_observed=true` 并监督进入时间；
- `observed_failure`：进入时间监督必须关闭，即使仿真在 taper miss 后又进入目标车道；
- `censored`：进入时间监督必须关闭并记录 censor reason。

诊断可以保存 `target_lane_entry_time_raw_s`，但不能把 failure/censored 的 raw late entry
重新解释为训练标签。

### Selector-v4 task actor contract

Selector-v4 不以 actor 当前 lane 直接推断“非冲突”。它对所有合法 ego candidate
commitment plans 构造 swept tubes，并与每个 actor 的 route-aware keep-lane 和所有可达相邻
lane hypotheses（含有界纵向加速度 envelope）求交。报告必须记录
`conflict_candidate_ids`、`earliest_conflict_time_s`、
`minimum_swept_obb_gap` 和 `conflict_surface_ids`。

`auxiliary_local` 只适用于配置的 auxiliary lane。main auxiliary edge 上的其他 lane 不因 edge
身份成为 critical；但 target front/rear、candidate conflict、nearest candidate conflict 和
conflict-eligible lowest TTC 始终保留。排序固定为 relevance class → mandatory role → earliest conflict time → TTC →
effective/surface gap → vehicle ID。vehicle ID 只作最终 tie-break。

coverage 分为两个职责：

- `task_actor_coverage_complete`：所有影响 Candidate task viability 的 critical actors 都进入
  ACCVP rows；它是全部 model splits/calibration/operating/test 的准入门槛；
- `risk_safety_actor_coverage_complete`：Risk/Shield 使用的全量当前车辆状态完整；这些车辆
  不必全部占用 ACCVP actor rows。

### Split and bootstrap

split component 至少连接以下关系：

```text
model_input_fingerprint
OR (scenario_route_hash, traffic_profile, episode_seed)
```

同 component 不得跨 train/validation/calibration/operating/test。ensemble bootstrap 按
component 抽样；同 fingerprint 的重复 root 作为一个等权组处理。

### Oracle cohort

seed 2/5 只验证历史 repairable premise：

```text
cohort_role = oracle_regression
root_policy = merge_timing
oracle_only = true
exclude_from_model_splits = true
required_seeds = [2, 5]
```

oracle 报告只有在 `oracle_state=go`、两个 seed 均有足够 deadline coverage、且完整 cohort
契约匹配时才有效。报告为 GO 但缺 `root_policy` 也必须视为不完整 artifact。

## 3. Canonical workflow

状态查询：

```powershell
python -m safe_rl.pipeline.run_accvp_vnext_pipeline
```

单阶段执行：

```powershell
python -m safe_rl.pipeline.run_accvp_vnext_pipeline --execute-next
```

门槛内连续执行：

```powershell
python -m safe_rl.pipeline.run_accvp_vnext_pipeline `
  --workflow-config safe_rl/config/active/accvp_vnext_selector4/workflow.yaml `
  --run-until pilot_validation
```

没有 `--run-until` 时，`--execute-next` 最多执行一个阶段。`--run-until` 只有在每个阶段
产物均通过 gate 后才继续；它不会忽略 fail report，也不会自动打开 final holdout。

阶段顺序为：

1. `selector_contract_audit`
2. `pilot_collection`
3. `pilot_merge`
4. `oracle_collection`
5. `oracle_merge`
6. `oracle_regression`
7. `pilot_validation`
8. `pilot_latency_feasibility_smoke`
9. `formal_collection`
10. `formal_merge`
11. `formal_validation`
12. `accvp_training`
13. `scorer_runtime_preflight`
14. `candidate_ppo_replicates`
15. `baseline_ppo_replicates`
16. `baseline_lineage_audit`
17. `policy_runtime_replicates`
18. `stage5_generate`
19. `stage5_replicates_and_aggregate`
20. `one_shot_final_holdout`

`selector_contract_audit` 首先验证旧 formal roots 的 Selector-v3 telemetry 是否覆盖每个 root
的全量 current actors；不完整时必须停止并做 selector-only root recollection。随后审计全部
5,000 roots、四种 Candidate 方法 × 五个 optimizer seeds 的 development states、历史 overflow
seeds 50021/50027/55027、原 runtime 捕获的不可变 overflow telemetry，以及
dense/aggressive/late-taper development stress states。历史 seed 的孤立重放不能替代原 telemetry，
因为复用环境中的 episode 顺序可能改变实际访问状态。容量
8/10/12 同时计算，8/10 只作为诊断；只有容量 12 达到零 overflow、零 mandatory drop、
100% protected coverage 且最大需求后至少保留 2 actors headroom 才能冻结。容量 12 失败则流程
blocked，不得回退 10 掩盖 runtime 或容量风险。

factorial orchestration 已接入该 workflow：它分别生成、审计和评估所有预注册
Reward/commitment 方法，并将最终方法绑定到 Reward-v3.1 + risk-gated commitment。阶段不再
人为 blocked；任何一个 child manifest、runtime 或 Stage5 lineage 不完整时仍会 fail closed。

`safe_rl.pipeline.run_full_pipeline` 是通用/历史 pipeline，不能替代以上 VNext 证据链。

## 4. Pilot gate

canonical Selector-v4 配置：

- `safe_rl/config/active/accvp_vnext_selector4/pilot.yaml`
- `safe_rl/config/active/accvp_vnext_selector4/oracle_regression.yaml`
- `safe_rl/config/active/accvp_vnext_selector4/formal.yaml`

pilot validation 同时检查：

- 每个预注册 collection source 的 root coverage；
- branch completion/success；
- observed viability fraction；
- schema/data contract 与 provenance；
- oracle dataset/report 的 seed、policy、cohort 和 split exclusion；
- critical overflow、rejected root、task/Risk coverage incomplete 均为 0；
- 全部 root/branch 的 actor-row mapping 与 root model-input fingerprint mismatch 均为 0；
- target front/rear、candidate conflict、nearest conflict、lowest conflict TTC protected actors
  覆盖率为 100%。

Pilot PASS 后、formal collection 前，workflow 会运行
`pilot_latency_feasibility_smoke`。它用 Pilot split 短训 1 epoch 的 12-actor 三成员 shadow
bundle，并在 66001--66005 development seeds 上执行真实 SUMO 的 ACCVP、Risk-secondary、
几何与 table packing 全路径。该报告固定为 `diagnostic_only_pre_formal_feasibility`，不能替代
60-seed 正式 scorer/policy runtime；但若 p95/p99/max 或 Risk p95 已明显超预算，会在耗时的
5,000-root formal collection 前 fail closed。
- pilot 数据本身没有混入 oracle seeds/rows。

当 `pilot_report.json` 为 `pilot_state=fail` 时，读取 `conditions`。如果仅
`oracle_regression=false`，应核对 oracle 报告的 `root_policy`、`cohort_role`、split exclusion
和 dataset provenance；不要重新采 pilot 或放宽 pilot threshold。

## 5. Interrupted collection recovery

schema3 collection shard 是不可变 artifact。配置使用：

```yaml
accvp:
  counterfactual:
    incomplete_shard_policy: quarantine
```

若同名 shard 只有部分文件或 manifest 不完整，collector 不覆盖原目录，而是移动到：

```text
shards/_failed_attempts/<utc-stamp>/
  quarantine_record.json
  ...original partial files...
```

随后才在原 shard path 进行干净重试。不要手工删除 `_failed_attempts`，除非完成独立 retention
审计并确认不再需要追溯故障。

## 6. ACCVP training and artifact lifecycle

formal training 的前置顺序是：配置 generation 校验、oracle 校验、dataset/schema/split
校验、formal runtime contract 校验，然后才初始化 Torch/CUDA 并构建模型。

正式 ensemble 至少三个 members。artifact bundle 使用：

```text
accvp_vnext_schema3_predictor.pt
accvp_vnext_schema3_calibration.json
accvp_vnext_schema3_candidate_manifest.json
```

manifest 应绑定 dataset/split/component、member seed、bootstrap hash、warm-start hash、最终
state-dict hash、calibration、operating point、runtime contract 和 observation feature hash。
candidate artifact 在 holdout promotion 前只允许 observation/shadow；active controller 必须有
相应 holdout GO。

Operating-point 的 `required_availability` 使用版本化分母
`risk_eligible_raw_or_merge_left_v1`：只在冻结 Risk authority 允许保留 raw action 或至少一个
merge-left rescue 的 decision 上衡量 ACCVP 模型 availability。报告必须同时保留 unconditional
candidate-set coverage 和 Risk-ineligible fraction；Risk-ineligible 状态始终 fail closed，不能因
提高 Risk threshold 或放宽 ACCVP gate 获得动作。

## 7. Reward factorial and PPO replication

`ppo_candidate_table_full.yaml` 是 replicate template，不是“唯一正式方法”。方法差异由
`ppo_ablation_matrix.yaml` 显式绑定：

```text
wcdt_reward_v2
candidate_table_reward_v2
candidate_table_reward_v2_commitment
candidate_table_reward_v3_1
candidate_table_reward_v3_1_commitment  <- final method candidate
```

Reward-v2 Candidate 回答 Candidate Table 本身是否有增益；Reward-v3.1 persistence 与
risk-gated commitment 回答识别机会后能否持续完成合流。每个正式方法至少五个独立 PPO
optimizer seeds；optimizer seed 与 simulator seed cohort 不得混用。

四个 Candidate 方法形成 Reward（v2/v3.1）× commitment（off/on）的 2×2 factorial，共
20 个 PPO checkpoints；WcDT Reward-v2 另有五个 baseline checkpoints。直接入口为：

```powershell
python -m safe_rl.pipeline.stage3_train_ppo_factorial `
  --config safe_rl/config/active/accvp_vnext_selector4/ppo_candidate_table_full.yaml `
  --matrix safe_rl/config/active/accvp_vnext_selector4/ppo_ablation_matrix.yaml `
  --workflow-config safe_rl/config/active/accvp_vnext_selector4/workflow.yaml `
  --optimizer-seeds 1001 1002 1003 1004 1005 `
  --output-root safe_rl_output/runs/accvp_vnext_selector4_factorial
```

`factorial_plan.json` 冻结全部 method/seed/config；每个训练完成后立即更新 child manifest。默认
resume 只复用 hash 和 Stage3 report 都匹配的结果。启动失败后遗留的纯空目录树可以自动
移除并重建；只要不完整 run 中已有任何文件或链接，协调器仍拒绝自动覆盖。
在 Windows 上，factorial manifest 的原子 `Path.replace` 若遇到病毒扫描器/索引器造成的瞬时
`PermissionError`，只执行五次以内的小幅退避重试；不会删除目标文件、覆盖部分 run 或绕过
lineage 校验。持续文件锁仍然 fail closed。

`run.seed` 是 simulator episode schedule 的起点，`rl.optimizer_seed` 是 PPO 网络与优化器
随机种子。由于 Stable-Baselines3 默认会把模型 seed 传播给 VecEnv，训练器会在模型创建后、
首次 rollout reset 前重新应用 `run.seed`。正式 preflight 与 Stage3 lineage 必须同时证明
simulator seeds 属于 `ppo_training`、optimizer seed 属于 `ppo_optimizer_replicates`。

生成配置和 checkpoints 只能写入 `safe_rl_output/runs`，manifest 至少记录 resolved config、
checkpoint、reward semantics、observation contract 和 ACCVP artifact fingerprint 的 hash。

## 8. Runtime claims

runtime 分两层：

1. scorer preflight：rule/base-state policy 访问真实 SUMO state，只测 ACCVP/Risk Candidate
   Table，不要求尚不存在的 159D Candidate PPO checkpoint；
2. policy runtime benchmark：四个 Candidate 方法的 20 个 PPO checkpoints 均测试完整
   observation/policy 路径；每个方法先取五副本最差值，总 gate 再要求四个方法全部通过。

synthetic fault audit 只证明 bounded-stale 状态机在 timeout、NaN、连续故障与恢复输入下的
行为。若报告声明 `synthetic_context=true`、`real_sumo_executed=false` 或
`hard_real_time_claim=false`，就不能据此声称 OS worker 可抢占或 hard real-time deployment。

bounded stale 状态不得通过 Risk gate；连续故障、恢复和 stale age 必须进入 observation 与
审计报告。

Selector-v4 的 scorer 热路径实现版本为
`accvp_runtime_conflict_selector_cached_geometry_v5`。性能修复限定为：单次 selector 调用内缓存
冻结的 scenario list/mapping、显式复制纯标量 `VehicleState`、candidate-conflict OBB 批处理只
返回 gap/overlap。selector parity 测试仍以 scalar reference 为权威，禁止为满足延迟而改变
selector/capacity/reward/commitment/Risk threshold。

旧 runtime implementation 的失败报告仅在 fingerprint 有效时归档，新实现不会复用其中的
episodes。同一 implementation、同一完整 seed request 的失败报告是不可变证据，coordinator
会显示具体 failed checks 并停止；只有实现发生实质修复并升级 version 后才能干净重跑。

## 9. Stage5 and holdout

同一 training seed 的左右方法必须使用完全相同的 simulator seed ledger。每个 replicate 先
生成 paired report，再按 training-seed × simulator-seed crossed bootstrap 聚合。六个比较分别
回答 Candidate Table、persistence、commitment 和最终方法效果。跨 Reward 比较不使用训练
`episode_reward`；crossed cells 不满足 pooled McNemar 的独立性时，报告明确不执行该检验。
只有存在显式且合法的同一家族 p 值时才执行 Holm；不会从置信区间反推 p 值。

VNext Stage5 使用嵌套的两级 confirmatory seed 预算：

- 五个次要机制比较使用 seed ledger 的固定前 100 个 seeds；
- `wcdt_v2_vs_final_method` 主要比较使用相同的前 100 个 seeds，并扩展到固定 300 个；
- 同一 method、optimizer seed、policy checkpoint、Risk checkpoint 和执行契约具有唯一 cache
  identity；单个 simulator seed 的 episode 记录创建后不可覆盖；
- 重复出现的方法从缓存读取，主要比较只执行尚未存在的后 200 个 seeds；
- paired report 和 replicated aggregate 都校验 cache identity、每个 episode 文件 hash、seed
  覆盖及 lineage，缓存不完整或被修改会 fail closed。

按五个 optimizer seeds 计算，分层设计若不复用仍需 8,000 episodes；不可变缓存把实际唯一
SUMO episodes 降为 4,500（五个唯一方法在 100-seed 前缀上各 500，加上主比较两种方法各扩展
1,000）。它替代原先“六个比较全部 300 seeds”的 18,000 episodes，但不减少主要比较的 300
seeds，也不改变训练、runtime 或 final holdout 的 seed cohort。

每个 Candidate group 必须绑定其自身 checkpoint 对应的 runtime preflight。最终 holdout 只接收
WcDT Reward-v2 vs Reward-v3.1+commitment 的标准 replicated child report，以及最终方法的五
副本 runtime child；factorial umbrella 用于上游完整性 gate，不直接替代这两个 promotion artifact。

冻结顺序：

```text
formal dataset
-> ACCVP bundle
-> scorer runtime
-> factorial method definitions
-> five PPO replicates per method
-> policy runtime
-> development and targeted development
-> code/config/threshold freeze
-> confirmatory cohorts
-> replicated aggregation
-> one-shot final holdout
```

final holdout 不允许生成阈值、选择 checkpoint、修改配置或重试以改善结论。完整 full artifact
仍是研究诊断路径；部署还需要 Lite artifact 的独立 promotion 和满足相应 hard-runtime 契约。

## 10. Failure triage

遇到 coordinator 停止时按以下顺序处理：

1. 运行无参数状态命令，确认第一个 incomplete phase；
2. 打开该 phase 的 artifact/report；
3. 找到唯一关闭的 `conditions`、`gate.pass` 或 lineage 字段；
4. 修复产生该字段的上游契约；
5. 重新运行同一个 `--run-until` 或 `--execute-next`；
6. 确认 `next_phase` 前移后再继续。

不要手工把 `fail` 改成 `pass`，不要复制其他 run 的报告，不要在 confirmatory/holdout 后调阈值。
