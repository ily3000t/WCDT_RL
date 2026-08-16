# Configuration registry and lifecycle

`registry.yaml` 是配置状态的唯一权威索引。不要根据文件名、静态引用次数或 YAML 是否能
加载来判断某个配置能否生成正式证据。

## Directory roles

```text
safe_rl/config/
  default_safe_rl.yaml
  registry.yaml
  active/
    accvp_vnext_selector3/ # canonical Selector-v3 entrypoints/workflow
    accvp_vnext/       # V1 diagnostic-only reproduction
    pipeline/          # maintained generic pipeline profiles
    smoke/             # maintained smoke-only profiles
  baselines/
    no_forecast/
    cv/
    wcdt/
  examples/            # templates; placeholders must be resolved
  archive/             # diagnostic_only historical reproduction
  local/               # gitignored machine-specific copies
```

Registry statuses：

- `canonical`：当前预注册的 ACCVP VNext 路径；
- `baseline`：维护中的比较组，不是 ACCVP candidate artifact；
- `supported`：维护中的 smoke/performance/operational 配置；
- `template`：示例，不是冻结实验；
- `diagnostic_only`：历史复现与 failure analysis，不能进入正式 lineage。

## `default_safe_rl.yaml` 的边界

`default_safe_rl.yaml` 是兼容默认值，不是可直接生成 VNext 正式证据的实验定义。字段按以下
原则维护：

- 默认关闭但代码仍支持的能力继续保留，例如 Shield、forecast features 和 ACCVP runtime；
- WcDT/Risk/通用 Stage1--5 所需的兼容字段继续保留，并在顶层区块注释其责任；
- 从未被运行时代码读取的开关已经移除。覆盖配置若再次传入这些键会立即报错，避免静默 no-op；
- 正式数据、Reward、runtime 和 holdout 语义只在
  `active/accvp_vnext_selector3/` 冻结；`active/accvp_vnext/` 已降级为
  `diagnostic_only`。

已退役 no-op 包括旧 `prediction.freeze/model_type/num_modes`、重复的
`forecast_features.normalize_features`、未生效的 feature include 开关、未生效的 Risk
architecture 开关，以及 Shield 的未生效排序开关。迁移时应使用报错中给出的有效字段；不要
为兼容旧 YAML 把这些键加回默认文件。

`accvp.data_contract_version` 没有被当作无用声明删除：collection 和 training 现在都会验证它
必须等于受支持的 schema3 data contract，并验证 dataset/config 一致。

## 设备分工

| 配置 | 作用 | VNext 建议 |
| --- | --- | --- |
| `training.stage2_device` | WcDT、Risk、ACCVP 批量训练 | 正式配置显式 `cuda:0` |
| `training.ppo_device` | 小型 PPO MLP 更新 | `cpu` |
| `training.forecast_runtime_device` | 在线 WcDT 推理 | `cpu`，保持 runtime contract |
| `training.diagnostics_device` | 大批量离线 diagnostics | `auto` 或显式 GPU |

ACCVP 只在 optimizer/validation batch 阶段使用 `stage2_device`。训练完成的 ensemble 会先回迁
CPU，再进行 calibration、operating-point selection、序列化和 runtime benchmark。因此改用
GPU 不会暗中改变在线设备契约，但会改变训练数值与 config lineage；已有 CPU artifact 不能被
重新标成 GPU artifact。

正式 deterministic CUDA 训练应在新进程启动前设置
`CUBLAS_WORKSPACE_CONFIG=:4096:8`。代码会关闭 TF32 并把实际训练设备写入 training history、
checkpoint metadata 和 manifest。

## SUMO 与 PPO 并行

| 配置 | 含义 | 默认值 |
| --- | --- | --- |
| `accvp.counterfactual.workers` | root collector 之外的 branch SUMO 进程数 | 2 |
| `accvp.counterfactual.pending_branch_jobs_per_worker` | 每个 branch worker 的队列回压深度 | 4 |
| `stage1.workers` | 通用 Risk-probe 的 SUMO worker 数 | 1 |
| `training.ppo_num_envs` | PPO rollout 的独立 SUMO/TraCI 进程数 | 1 |
| `training.ppo_worker_torch_threads` | 每个 PPO environment worker 的 Torch 线程数 | 1 |
| `training.ppo_main_torch_threads` | PPO 主进程更新网络的 Torch 线程数 | 4 |
| `training.ppo_expected_rollout_size` | 可选的 `num_envs × n_steps` 语义门槛 | null |
| `training.max_parallel_optimizer_replicates` | factorial 父进程同时运行的独立 PPO replicates，限定 1 或 2 | 1 |
| `stage3.checkpoint_selection_workers` | 每个 checkpoint 的独立 selection-seed SUMO workers | 1 |
| `stage3.checkpoint_selection_worker_torch_threads` | 每个 selection worker 的 Torch 线程数 | 1 |

Canonical Candidate PPO 把 `ppo_expected_rollout_size` 固定为 1024。若从 1 个环境改为 2 个，
必须同时把 `rl.n_steps` 从 1024 改为 512；4 个环境对应 256。否则程序会在启动 SUMO rollout
前拒绝配置。每个 PPO worker 都独立加载 ACCVP/Risk checkpoint，因此先测试 2 个环境，并用
内存、TraCI 稳定性和 steps/s 报告决定是否增加，不要把单纯增加进程数当作等价加速。

Checkpoint selection 的 worker 只生成逐 seed episode report；父进程按 seed 排序后统一聚合、
评分和写 best-checkpoint 证据。`simulation_blocking_exact_v1` 下 CPU 竞争不会改变 observation，
但并行运行产生的 wall-clock latency telemetry 不是 deployment runtime 证据。正式 Selector-v4
配置使用 4 个 selection workers，selection seeds、checkpoint 数量和评分公式不变。

Optimizer replicate 外层并行必须使用独立 spawn 子进程，factorial/child manifests 仍只由父进程
写入。能力上限固定为 2；在完成同工作量的 `1 replicate × 4 envs` 与
`2 replicates × 2 envs` 本机基准前，正式配置保持 1。

反事实 formal collection 的 `workers` 和 `shard_roots` 属于已生成数据的采集 lineage。不要
为了应用新默认值重跑已完成 shard，也不要在正在执行的 collection 中修改它们。

`advanced/` 平铺目录已退役。旧 basename 保存在相应 archive family 中；canonical VNext 使用
简短角色名。

## Canonical Selector-v3 set

```text
active/accvp_vnext_selector3/selector_audit.yaml
active/accvp_vnext_selector3/workflow.yaml
active/accvp_vnext_selector3/pilot.yaml
active/accvp_vnext_selector3/oracle_regression.yaml
active/accvp_vnext_selector3/formal.yaml
active/accvp_vnext_selector3/train.yaml
active/accvp_vnext_selector3/ppo_candidate_table_full.yaml
active/accvp_vnext_selector3/ppo_ablation_matrix.yaml
```

`selector_audit.yaml` 是唯一允许在容量报告产生前加载 Selector-v3 的配置。
`workflow.yaml` 描述阶段顺序、artifact gate、最终方法角色和六个 factorial 比较；数值科学
阈值仍保留在各自 pilot/runtime/Stage5 配置中，避免重复声明和漂移。

## Reward and method binding

Candidate PPO template 本身不定义“最终方法”。正式方法必须通过
`ppo_ablation_matrix.yaml` 绑定 Reward 与 commitment 因素：

| method_id | 正式角色 |
| --- | --- |
| `wcdt_reward_v2` | WcDT baseline |
| `candidate_table_reward_v2` | Candidate Table 单因素消融 |
| `candidate_table_reward_v2_commitment` | Reward-v2 下的 commitment 对照 |
| `candidate_table_reward_v3_1` | persistence reward 对照 |
| `candidate_table_reward_v3_1_commitment` | 最终完整方法候选 |

Reward-v2 Candidate 必须保留以完成因果归因，但不能沿 pipeline 被当作唯一主方法送入
Stage5/holdout。factorial 协调器现在固定生成四个 Candidate 方法 × 五个 optimizer seeds，
总 manifest 只有在 20 个 checkpoints、四组语义和最终方法角色全部验证后才会标记 complete。
`ppo_ablation_matrix.yaml` 因此属于 canonical 冻结协议，而不是可随意修改的辅助模板。

正式 runtime 同样覆盖四个 Candidate 方法，而不是用最终方法的一份 preflight 代替其他
checkpoint。Stage5 配置按 group 绑定 policy-runtime report；跨 Reward 版本只比较共同任务与
安全指标，不比较各自训练时定义不同的 `episode_reward`。

## Running the registry workflow

只查看状态：

```powershell
python -m safe_rl.pipeline.run_accvp_vnext_pipeline
```

执行一个阶段：

```powershell
python -m safe_rl.pipeline.run_accvp_vnext_pipeline --execute-next
```

连续执行到一个已命名门槛：

```powershell
python -m safe_rl.pipeline.run_accvp_vnext_pipeline `
  --workflow-config safe_rl/config/active/accvp_vnext_selector3/workflow.yaml `
  --run-until pilot_validation
```

例如 scorer preflight 已完成后，可以只推进 factorial PPO：

```powershell
python -m safe_rl.pipeline.run_accvp_vnext_pipeline `
  --run-until candidate_ppo_replicates
```

该阶段默认可恢复。已完整且 hash 匹配的 replicate 会跳过；不完整 run 目录不会被自动覆盖。

命令会在子进程失败、产物 gate 关闭、workflow blocked 或目标阶段完成时停止。final holdout
还要求独立的 `--allow-final-holdout`。`safe_rl.pipeline.run_full_pipeline` 是通用/历史比较
协调器，不是 VNext 正式入口。

## Generated configs and local paths

PPO factorial、runtime、Stage5 paired config 和 resolved config 写入：

```text
safe_rl_output/runs/accvp_vnext_selector3_factorial/
safe_rl_output/runs/accvp_vnext_selector3_runtime/
safe_rl_output/runs/accvp_vnext_selector3_stage5/generated/
```

不要把每个 optimizer seed 的生成配置加入 `active/`，否则会重新造成配置膨胀。机器专用路径
应从 example 复制到 `local/`；`local/` 不跟踪，也不能作为正式证据来源。

不要移动已有 `safe_rl_output/runs` 目录来匹配配置分类。报告和 manifest 绑定原路径/hash；
retention 或删除应在独立 manifest-driven audit 后进行。

## Archive policy

archive 配置仍有三类价值：历史实验复现、方法演进说明、failure-case regression。它们不得
用于 VNext training、calibration、operating-point selection、confirmatory evaluation、final
holdout 或 artifact promotion。`artifact_revocation_manifest_selector3.json` 将 schema2 和
Selector-v2 V1 artifacts 标为 `diagnostic_only`；Selector-v3 正式 lineage 不得复用这些
产物。registry family coverage 由测试强制检查。
