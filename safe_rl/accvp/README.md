# ACCVP VNext protocol and execution guide

本文档是 ACCVP VNext 的详细执行契约。日常入口见仓库根 `README.md`；配置状态以
`safe_rl/config/registry.yaml` 为准；机器生成的 artifact/report 才决定阶段 gate 是否打开。

## 1. Compatibility boundary

VNext 正式数据和模型必须同时满足：

- `artifact_generation: vnext_schema3`
- `schema_version: 3`
- `data_contract_version: accvp_240_v2`
- `actor_row_mapping_version: selected_indices_v2`
- `root_observation_fingerprint_version: model_input_fingerprint_v3`
- `entry_time_label_version: conditional_entry_time_v1`
- `loss_version: accvp_loss_v2`

旧 schema2 数据没有足够证据证明 `selected_actor_ids`、branch response rows 与 WcDT token
rows 的唯一映射，因此不能迁移为完整 VNext actor-response supervision。它只允许在明确
标注的 diagnostic/migration audit 中使用。

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
  --workflow-config safe_rl/config/active/accvp_vnext/workflow.yaml `
  --run-until pilot_validation
```

没有 `--run-until` 时，`--execute-next` 最多执行一个阶段。`--run-until` 只有在每个阶段
产物均通过 gate 后才继续；它不会忽略 fail report，也不会自动打开 final holdout。

阶段顺序为：

1. `pilot_collection`
2. `pilot_merge`
3. `oracle_collection`
4. `oracle_merge`
5. `oracle_regression`
6. `pilot_validation`
7. `formal_collection`
8. `formal_merge`
9. `accvp_training`
10. `scorer_runtime_preflight`
11. `candidate_ppo_replicates`
12. `baseline_ppo_replicates`
13. `policy_runtime_replicates`
14. `stage5_generate`
15. `stage5_replicates_and_aggregate`
16. `one_shot_final_holdout`

当前 workflow 故意将 `candidate_ppo_replicates` 标记为 blocked。解锁条件不是“某个 PPO 能
运行”，而是 factorial orchestration 已能分别生成、审计和评估所有预注册 Reward/commitment
方法，并将最终方法绑定到 Reward-v3.1 + risk-gated commitment。

`safe_rl.pipeline.run_full_pipeline` 是通用/历史 pipeline，不能替代以上 VNext 证据链。

## 4. Pilot gate

canonical 配置：

- `safe_rl/config/active/accvp_vnext/pilot.yaml`
- `safe_rl/config/active/accvp_vnext/oracle_regression.yaml`
- `safe_rl/config/active/accvp_vnext/formal.yaml`

pilot validation 同时检查：

- 每个预注册 collection source 的 root coverage；
- branch completion/success；
- observed viability fraction；
- schema/data contract 与 provenance；
- oracle dataset/report 的 seed、policy、cohort 和 split exclusion；
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

生成配置和 checkpoints 只能写入 `safe_rl_output/runs`，manifest 至少记录 resolved config、
checkpoint、reward semantics、observation contract 和 ACCVP artifact fingerprint 的 hash。

## 8. Runtime claims

runtime 分两层：

1. scorer preflight：rule/base-state policy 访问真实 SUMO state，只测 ACCVP/Risk Candidate
   Table，不要求尚不存在的 159D Candidate PPO checkpoint；
2. policy runtime benchmark：五个 Candidate PPO 训练后测试完整 observation/policy 路径，
   聚合采用最差 replicate，而非平均值。

synthetic fault audit 只证明 bounded-stale 状态机在 timeout、NaN、连续故障与恢复输入下的
行为。若报告声明 `synthetic_context=true`、`real_sumo_executed=false` 或
`hard_real_time_claim=false`，就不能据此声称 OS worker 可抢占或 hard real-time deployment。

bounded stale 状态不得通过 Risk gate；连续故障、恢复和 stale age 必须进入 observation 与
审计报告。

## 9. Stage5 and holdout

同一 training seed 的 baseline/candidate 必须使用完全相同的 simulator seed ledger。每个
replicate 先生成 paired report，再跨 optimizer seeds 做 hierarchical aggregation。正式统计
至少包含 paired BCa、McNemar、多重比较校正和完整 lineage。

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
