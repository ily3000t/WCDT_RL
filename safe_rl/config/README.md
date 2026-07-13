# Configuration registry and lifecycle

`registry.yaml` 是配置状态的唯一权威索引。不要根据文件名、静态引用次数或 YAML 是否能
加载来判断某个配置能否生成正式证据。

## Directory roles

```text
safe_rl/config/
  default_safe_rl.yaml
  registry.yaml
  active/
    accvp_vnext/       # canonical VNext entrypoints and workflow contract
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

`advanced/` 平铺目录已退役。旧 basename 保存在相应 archive family 中；canonical VNext 使用
简短角色名。

## Canonical VNext set

```text
active/accvp_vnext/workflow.yaml
active/accvp_vnext/pilot.yaml
active/accvp_vnext/oracle_regression.yaml
active/accvp_vnext/formal.yaml
active/accvp_vnext/train.yaml
active/accvp_vnext/ppo_candidate_table_dev.yaml
active/accvp_vnext/ppo_candidate_table_full.yaml
active/accvp_vnext/ppo_ablation_matrix.yaml
```

`workflow.yaml` 描述阶段顺序、artifact gate、blocked phase 和最终方法角色；数值科学阈值仍
保留在各自 pilot/runtime/Stage5 配置中，避免重复声明和漂移。

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
Stage5/holdout。当前 workflow 因此在 factorial PPO orchestration 完成前阻塞
`candidate_ppo_replicates`。

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
  --workflow-config safe_rl/config/active/accvp_vnext/workflow.yaml `
  --run-until pilot_validation
```

命令会在子进程失败、产物 gate 关闭、workflow blocked 或目标阶段完成时停止。final holdout
还要求独立的 `--allow-final-holdout`。`safe_rl.pipeline.run_full_pipeline` 是通用/历史比较
协调器，不是 VNext 正式入口。

## Generated configs and local paths

PPO replicate、Stage5 paired config 和 resolved config 写入：

```text
safe_rl_output/runs/<run>/generated_configs/
```

不要把每个 optimizer seed 的生成配置加入 `active/`，否则会重新造成配置膨胀。机器专用路径
应从 example 复制到 `local/`；`local/` 不跟踪，也不能作为正式证据来源。

不要移动已有 `safe_rl_output/runs` 目录来匹配配置分类。报告和 manifest 绑定原路径/hash；
retention 或删除应在独立 manifest-driven audit 后进行。

## Archive policy

archive 配置仍有三类价值：历史实验复现、方法演进说明、failure-case regression。它们不得
用于 VNext training、calibration、operating-point selection、confirmatory evaluation、final
holdout 或 artifact promotion。`artifact_revocation_manifest_vnext.json` 是旧 artifact 的撤销
边界；registry family coverage 由测试强制检查。
