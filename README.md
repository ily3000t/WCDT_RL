# WcDT_RL：基于 WcDT 预测的 SAFE_RL 高速汇入实验

本仓库在原始 WcDT 交通场景生成代码基础上，新增了 `safe_rl/` 包，用于构建 SUMO `highway_merge` 场景中的分层安全强化学习框架。

核心目标不是“用 WcDT 替代强化学习决策”，而是：

```text
SUMO highway_merge 场景
  -> WcDT-style 未来交通预测
  -> Risk Module 候选动作风险评估
  -> PPO 学习驾驶策略
  -> Shield 在高风险动作出现时替换动作
  -> paired evaluation 验证闭环安全收益
```

## 目录结构

```text
safe_rl/
  config/default_safe_rl.yaml
  sim/                 # SUMO 环境、动作空间、风险指标、场景校验
  prediction/          # SUMO -> WcDT 适配、预测特征增强
  risk/                # 显式风险特征、学习型 Risk Module、候选动作排序
  shield/              # Shield 动作替换与 fallback policy
  rl/                  # SB3 PPO 训练与评估封装
  pipeline/            # Stage1-Stage5 命令入口

scenarios/highway_merge/
  highway_merge.sumocfg
  highway_merge.rou.xml
  highway_merge.net.xml
```

原始 WcDT 模型代码仍保留在 `net_works/` 下；`BackBone` 额外新增了 `predict()` 推理接口，原训练 `forward()` 未改变。

## 环境准备

推荐使用 `environment.yml` 创建环境：

```powershell
conda env create -f environment.yml
conda activate WcDT
```

需要安装 SUMO，并保证 `sumo`、`netconvert` 在 `PATH` 中。当前代码也会自动尝试从 `SUMO_HOME/tools` 和 `sumo.exe` 同级目录推导 TraCI 路径。

Windows PowerShell 示例：

```powershell
$env:SUMO_HOME = "E:\Program Files\sumo-1.22.0"
sumo --version
netconvert --version
```

如果只验证场景是否能加载：

```powershell
python scenarios\highway_merge\build_network.py
sumo -c scenarios\highway_merge\highway_merge.sumocfg --end 1 --no-step-log true --duration-log.disable true --seed 1
```

## 配置文件

默认配置：

```text
safe_rl/config/default_safe_rl.yaml
```

所有 Stage 默认都会加载该配置；如需覆盖配置，传入：

```powershell
--config path\to\your_config.yaml
```

最重要的实验开关：

```yaml
forecast_features:
  enabled: false

rl:
  use_wcdt_forecast_features: false

shield:
  enabled: false
```

含义：

```text
forecast_features=false, shield=false  -> 原始 PPO
forecast_features=true,  shield=false  -> PPO + forecast 预测特征（constant_velocity 或 WcDT）
forecast_features=false, shield=true   -> PPO + Shield
forecast_features=true,  shield=true   -> PPO + forecast 预测特征 + Shield
```

## 五阶段命令

建议显式指定同一个 `RUN_ID`，这样 Stage2-Stage5 会使用同一轮实验输出。

PowerShell：

```powershell
$RUN_ID = "safe_rl_highway_merge_001"
```

### Stage1：SUMO 危险先验采集

采集 highway_merge 场景中的仿真危险先验、显式风险指标、历史/未来轨迹窗口。

```powershell
python -m safe_rl.pipeline.stage1_risk_probe --run-id $RUN_ID
```

主要输出：

```text
safe_rl_output/runs/<run_id>/stage1/risk_probe_buffer.npz
safe_rl_output/runs/<run_id>/stage1/risk_events.jsonl
safe_rl_output/runs/<run_id>/stage1/stage1_report.json
safe_rl_output/runs/<run_id>/stage1/audit/stage1_data_audit.json
safe_rl_output/runs/<run_id>/stage1/audit/stage1_action_histogram.csv
safe_rl_output/runs/<run_id>/stage1/audit/stage1_action_histogram.png
safe_rl_output/runs/<run_id>/stage1/audit/stage1_reward_distribution.png
safe_rl_output/runs/<run_id>/stage1/audit/stage1_risk_distribution.png
safe_rl_output/runs/<run_id>/stage1/replay/episode_0000.json
safe_rl_output/runs/<run_id>/stage1/tensorboard/
```

命令行会显示：

```text
[stage1] run_id=...
[stage1] SUMO config=...
[stage1] SUMO binary=sumo, episodes=...
Stage1 episodes:  50%|...
[stage1] episode=0 replay=...
[stage1] audit=...
[stage1] buffer=...
```

Stage1 默认使用 mixed sampler：少量随机动作、主要 merge heuristic、部分 risk-seek 动作。风险 buffer 会对每个状态展开 9 个候选动作样本，风险标签重点关注 ego 从匝道汇入目标主路 lane 2 的 front/rear gap、ramp 局部 gap、merge-zone risk。审计会统计执行动作分布、候选动作风险分布、per-action risk rate、target-lane gap 分位数、risk type rate、reward 分布和 trajectory sample 数量。

### Stage2：WcDT-style Prediction + Risk Module 训练

使用 Stage1 buffer 训练风险模块；如果 buffer 中有轨迹窗口，也会训练 WcDT-style predictor。

```powershell
python -m safe_rl.pipeline.stage2_train_prediction_risk --run-id $RUN_ID
```

主要输出：

```text
safe_rl_output/runs/<run_id>/stage2/wcdt_predictor.pt
safe_rl_output/runs/<run_id>/stage2/wcdt_predictor_best.pt
safe_rl_output/runs/<run_id>/stage2/wcdt_v2_predictor.pt
safe_rl_output/runs/<run_id>/stage2/wcdt_v2_predictor_best.pt
safe_rl_output/runs/<run_id>/stage2/risk_module.pt
safe_rl_output/runs/<run_id>/stage2/stage2_initial_prediction_report.json
safe_rl_output/runs/<run_id>/stage2/stage2_training_report.json
safe_rl_output/runs/<run_id>/stage2/tensorboard/
```

Stage2 的 WcDT-style predictor 会从 Stage1 trajectory windows 中划分 train/validation，按 validation `FDE + 0.5 * target_lane_gap_abs_error + 0.5 * future_min_distance_abs_error` 选择 best checkpoint。`wcdt_predictor_best.pt` 保存最佳权重；兼容路径 `wcdt_predictor.pt` 也写入同一 best 权重，避免后续 Stage3/Stage5 加载最后一个退化 epoch。

Stage2 还会训练独立的 `wcdt_v2` residual-over-CV predictor。它不覆盖旧 WcDT，输入更偏 merge-centric：target lane 2 front/rear、ramp front/rear 和 nearest conflict vehicle。默认保存 3-model ensemble，`uncertainty` 来自 ensemble 方差。Risk Module validation 报告新增 legal candidate 上的 `ECE / Brier / NLL / reliability_bins`，并计算可选 temperature scaling 诊断；默认不把 calibration 写入 runtime，只有显式开启配置后 Shield 才使用 calibrated score。

说明：当前仓库没有预训练 WcDT checkpoint，因此默认从 SUMO 采集数据训练；如有外部权重，可在配置中设置：

```yaml
prediction:
  checkpoint: "path/to/wcdt_checkpoint.pt"
```

Stage4 采集完成后，如果要把 on-policy failure buffer 合并回 Risk Module 训练，使用：

```powershell
python -m safe_rl.pipeline.stage2_train_prediction_risk --run-id $RUN_ID --config safe_rl\config\advanced\stage2_with_stage4.yaml
```

该配置会读取同一 run 下的 `stage1/risk_probe_buffer.npz` 和 `stage4/on_policy_failure_buffer.npz`，Risk Module 使用两者拼接后的风险样本训练；WcDT predictor 仍优先使用 Stage1 的 trajectory windows。

### Stage3：PPO 强化学习训练

训练 highway_merge ego agent。默认是不带 WcDT forecast features 的 PPO。

```powershell
python -m safe_rl.pipeline.stage3_train_ppo --run-id $RUN_ID
```

主要输出：

```text
safe_rl_output/runs/<run_id>/stage3/ppo_model.zip
safe_rl_output/runs/<run_id>/stage3/stage3_training_report.json
safe_rl_output/runs/<run_id>/stage3/tensorboard/
```

Stage3 使用 Stable-Baselines3 的 TensorBoard 记录 PPO reward、episode length、policy loss、value loss、entropy 等训练指标。

如果要先训练不依赖 checkpoint 的 `PPO + constant-velocity forecast features`，使用覆盖配置：

```text
safe_rl/config/advanced/ppo_constant_velocity_features.yaml
```

如果要训练旧版 `PPO + WcDT v1 forecast features`，使用历史兼容覆盖配置：

```text
safe_rl/config/advanced/ppo_forecast_features.yaml
safe_rl/config/advanced/ppo_wcdt_v1_features_legacy.yaml
```

`ppo_forecast_features.yaml` 保留是为了旧命令不失效；新主线推荐通过 full runner 生成 CV + WcDT v2 的 forecast PPO 配置。注意：forecast-feature PPO 是 63 维 observation，不能复用 baseline 的 52 维 PPO；如果手动训练 WcDT v1，需要把配置中的 `forecast_features.checkpoint` 指向 baseline run 的 `stage2/wcdt_predictor.pt`。

```powershell
$FORECAST_RUN_ID = "safe_rl_highway_merge_forecast_001"
python -m safe_rl.pipeline.stage3_train_ppo --run-id $FORECAST_RUN_ID --config safe_rl\config\advanced\ppo_forecast_features.yaml
```

### Stage4：RL 过程危险片段采集与风险模型修正数据

默认使用 shadow mode：不真正替换 PPO 动作，只记录 Shield 会如何干预。

```powershell
python -m safe_rl.pipeline.stage4_collect_failures --run-id $RUN_ID
```

主要输出：

```text
safe_rl_output/runs/<run_id>/stage4/on_policy_failure_buffer.npz
safe_rl_output/runs/<run_id>/stage4/intervention_buffer.jsonl
safe_rl_output/runs/<run_id>/stage4/stage4_report.json
safe_rl_output/runs/<run_id>/stage4/replay/episode_0000.json
safe_rl_output/runs/<run_id>/stage4/tensorboard/
```

如果要启用真实 intervention mode，使用覆盖配置 `safe_rl/config/advanced/stage4_intervention.yaml`。

命令：

```powershell
python -m safe_rl.pipeline.stage4_collect_failures --run-id $RUN_ID --config safe_rl\config\advanced\stage4_intervention.yaml
```

Stage4 report 会额外记录 action histogram、shadow would-replace rate、fallback rate、raw risk 分布和 replacement risk delta，用于判断 Shield 是否仍在过度干预。

### Stage5：Shield on/off 成对闭环安全评估

默认命令只比较 baseline 52 维 PPO 的 `ppo` 和 `ppo_shield` 两组，使用相同 SUMO seed、相同 PPO checkpoint、相同初始交通状态，只改变 Shield 开关做 paired evaluation。

```powershell
python -m safe_rl.pipeline.stage5_paired_eval --run-id $RUN_ID
```

主要输出：

```text
safe_rl_output/runs/<run_id>/stage5/formal_paired_eval_report.json
safe_rl_output/runs/<run_id>/stage5/shield_off_metrics.json
safe_rl_output/runs/<run_id>/stage5/shield_on_metrics.json
safe_rl_output/runs/<run_id>/stage5/replay/<group>_seed_<seed>.json
safe_rl_output/runs/<run_id>/stage5/tensorboard/
```

如果要手动比较 baseline、CV forecast 和 WcDT v2 forecast，建议分别训练 baseline PPO、CV forecast PPO、WcDT v2 forecast PPO，然后复制并修改当前主线模板中的 `model_path`：

```text
safe_rl/config/advanced/stage5_six_groups_cv_wcdt_v2.example.yaml
```

旧模板 `safe_rl/config/advanced/stage5_four_groups.example.yaml` 是 WcDT v1 legacy 示例，不作为新主实验模板。模板中的 forecast 组必须显式设置对应 63 维 forecast PPO 的 `model_path`。CV、WcDT v1 和 WcDT v2 虽然 observation 都是 63 维，但 forecast feature 分布不同，应分别训练 PPO。`forecast_source: "wcdt"` 还要设置 `forecast_checkpoint` 指向 Stage2 的 `wcdt_predictor.pt`；`forecast_source: "wcdt_v2"` 指向 `wcdt_v2_predictor.pt`；`forecast_source: "constant_velocity"` 不需要 checkpoint。Stage5 会在评估前校验 PPO model 和环境 observation shape，不匹配会直接失败。

命令：

```powershell
python -m safe_rl.pipeline.stage5_paired_eval --run-id $RUN_ID --config path\to\your_stage5_six_groups.yaml
```

### Stage5 Shield 阈值扫描

当已有完整 run 输出后，可以不重训 Stage1-Stage4，直接复用当前 PPO、forecast PPO 和 `stage2/risk_module.pt` 扫描 Shield 阈值：

```powershell
python -m safe_rl.pipeline.stage5_shield_sweep --run-id $RUN_ID
```

默认扫描 `activation_risk_threshold/replacement_margin` 的四组组合：`0.90/0.15`、`0.85/0.15`、`0.85/0.10`、`0.80/0.10`。输出写入：

```text
safe_rl_output/runs/<run_id>/stage5_sweep/shield_sweep_report.json
safe_rl_output/runs/<run_id>/stage5_sweep/generated_configs/
```

报告会按 variant 汇总 reward、min distance、TTC、DRAC、真实 replacement、fallback 和 regression 检查，并给出 `recommended_variant`。同时会输出 Shield score 饱和诊断，包括 raw risk score、best candidate risk score、replacement risk delta、reason ratio 和 raw risk 到 activation threshold 的 margin 分布，用来判断为什么不同阈值组合可能产生完全相同的动作。

默认阈值仍保持保守设置：

```yaml
shield:
  activation_risk_threshold: 0.90
  replacement_margin: 0.15
  allow_fallback: false
  emergency_fallback_enabled: true
```

`emergency_fallback` 是独立的极端物理风险兜底，只在 `min_ttc <= 0.30`、`min_distance <= 1.0`，或 raw/best candidate 风险都饱和且进入 watch 区时触发。它优先选择合法的 `keep_decelerate` / `keep_hold`，并单独记录为 `emergency_fallback_count`，不会混入普通 `fallback_count`。

如果只想做诊断，可以额外扫描更激进阈值：

```powershell
python -m safe_rl.pipeline.stage5_shield_sweep --run-id $RUN_ID --include-aggressive
```

如需对比 raw risk score 与 temperature-scaled score 对 Shield 行为的影响，增加：

```powershell
python -m safe_rl.pipeline.stage5_shield_sweep --run-id $RUN_ID --include-calibrated
```

`--include-aggressive` 和 `--include-calibrated` 只用于解释风险分数饱和、校准效果和阈值敏感性，不会自动改写默认 Shield 配置。sweep report 会输出 `calibration_effect_summary` 和 `threshold_sensitivity_summary`，用于判断 calibrated score 是否真正改变替换率、是否造成 regression、不同阈值下替换率是否可解释变化。当前默认 Shield 仍保持 `0.90/0.15`。

## 一键顺序运行示例

当前推荐实验顺序只包含三步：full pipeline、50-seed confirmatory、calibrated sweep。100-seed 复验、分层 confirmatory 和提高场景难度放到模型链路完全验证之后再做。

推荐直接使用全流程 runner。它会重建网络、做 SUMO smoke check、依次运行 Stage1/2/3/4、用 Stage4 buffer 重训 Risk Module，然后在同一份 Stage1/Stage4 数据、同一个 Risk Module、同一个 baseline PPO 上分别训练 CV forecast PPO 和 WcDT v2 forecast PPO，并完成多组 Stage5 paired evaluation：

```powershell
python -m safe_rl.pipeline.run_full_pipeline --run-id safe_rl_wcdt_v2_002 --forecast-sources constant_velocity,wcdt_v2
python -m safe_rl.pipeline.stage5_confirmatory_eval --run-id safe_rl_wcdt_v2_002 --episodes 50
python -m safe_rl.pipeline.stage5_shield_sweep --run-id safe_rl_wcdt_v2_002 --include-calibrated
```

默认 `--forecast-sources` 是 `constant_velocity,wcdt_v2`。旧 WcDT v1 不再默认运行；如果要保留 v1 对照，需要显式运行 `constant_velocity,wcdt` 或 `constant_velocity,wcdt,wcdt_v2`。如果只想跑其中一个 forecast 分支：

```powershell
python -m safe_rl.pipeline.run_full_pipeline --run-id safe_rl_merge_local_cv_001 --forecast-sources constant_velocity
python -m safe_rl.pipeline.run_full_pipeline --run-id safe_rl_merge_local_wcdt_001 --forecast-sources wcdt
python -m safe_rl.pipeline.run_full_pipeline --run-id safe_rl_merge_local_wcdt_v2_001 --forecast-sources wcdt_v2
```

旧参数 `--forecast-source wcdt` 仍可用于单分支兼容，但不要和 `--forecast-sources` 同时使用。

如需覆盖采样轮数和 PPO 训练步数：

```powershell
python -m safe_rl.pipeline.run_full_pipeline --run-id safe_rl_merge_local_001 --stage1-episodes 500 --ppo-timesteps 20000 --forecast-sources constant_velocity,wcdt_v2
```

生成的临时配置会写入：

```text
safe_rl_output/runs/<run_id>/generated_configs/
safe_rl_output/runs/<run_id>/generated_configs/forecast_cv_ppo.yaml
safe_rl_output/runs/<run_id>/generated_configs/forecast_wcdt_v2_ppo.yaml
safe_rl_output/runs/<run_id>/generated_configs/stage5_multi_groups.yaml
safe_rl_output/runs/<run_id>/stage5/diagnostics/forecast_diagnostics.json
```

只有显式包含 `wcdt` 时才会生成 `forecast_wcdt_ppo.yaml`。当前主线实验可以按三条轴线理解，避免把名字混在一起：

```text
PPO training reward: default / safety_forecast / shield_guided_forecast
PPO observation: 52D baseline / 63D CV / 63D WcDT v1 legacy / 63D WcDT v2
Eval-time Shield: off / on
```

`safety_forecast` 和 `shield_guided_forecast` 都只是普通 PPO 的 reward profile；`ppo_shield` 是同一个 PPO 在评估时套 eval-time Shield，不是重新训练的模型。

完成 full pipeline 后，推荐用 50 seeds 做最终复验。该命令只复用已有 Stage1/2/3/4 输出，不重训模型：

```powershell
python -m safe_rl.pipeline.stage5_confirmatory_eval --run-id safe_rl_merge_local_001
```

短流程 smoke 可以先跑：

```powershell
python -m safe_rl.pipeline.stage5_confirmatory_eval --run-id safe_rl_merge_local_001 --episodes 5
```

confirmatory 输出：

```text
safe_rl_output/runs/<run_id>/stage5_confirmatory/formal_paired_eval_report.json
safe_rl_output/runs/<run_id>/stage5_confirmatory/confirmatory_summary.json
safe_rl_output/runs/<run_id>/stage5_confirmatory/generated_configs/stage5_confirmatory.yaml
```

`confirmatory_summary.json` 会把主结果明确分成：可信 Shield 主线 `ppo/ppo_shield`、forecast 对照 `ppo_cv_features`、当前推荐预测分支 `ppo_wcdt_v2_features`、当前最强安全组合 `wcdt_v2_prediction_shield`。报告还会写入 `model_role_explanations` 和 `reporting_recommendation`，方便直接整理结果表。旧 WcDT v1 的 `ppo_wcdt_features` / `wcdt_prediction_shield` 不删除，但只作为 ablation/diagnostic，不作为最终有效预测模块结论。

如果 `wcdt_v2_prediction_shield` 没有实际替换但不退化，会标记 `shield_not_needed_on_wcdt_v2_policy=true`；如果只发生少量替换但安全指标不退化，会标记 `low_frequency_safety_backstop=true`，表示 Shield 在 WcDT v2 policy 上作为低频补强，而不是失败。

主结果表建议同时报告四类指标：

```text
Safety: collision_rate, near_miss_rate, proxy_collision_rate, safety_violation_rate, proxy_collision_count, safety_violation_count, min_distance_le_collision_threshold_count, min_distance_p1, ttc_p1, drac_p99_capped
Task/Efficiency: merge_success_rate, completion_time_mean, completion_time_p95
Driving Behavior: ego_speed_mean, ego_speed_p10, hard_brake_rate
Shield: mean_actual_replacements, actual_replacement_rate, fallback_rate, emergency_fallback_count, emergency_fallback_rate
```

报告仍保留 `drac_p99_raw` 和兼容字段 `drac_p99`，用于 debug `minD=0 / DRAC=1e6` 这类极端异常；主表优先使用 capped 后的 `drac_p99_capped`。

补充效率和舒适性指标后，不需要重跑 Stage1/2/3/4；只需要重跑 Stage5 confirmatory 和 calibrated sweep 即可刷新报告。

Forecast feature 利用率诊断建议按以下顺序运行。该流程不改变 63 维 observation，只修正 `forecast_merge_gap` 的语义并检查 PPO 是否真的使用 forecast 维度：

```powershell
python -m safe_rl.pipeline.run_full_pipeline --run-id safe_rl_wcdt_v2_featurefix_001 --forecast-sources constant_velocity,wcdt_v2
python -m safe_rl.pipeline.stage5_confirmatory_eval --run-id safe_rl_wcdt_v2_featurefix_001 --episodes 50
python -m safe_rl.pipeline.forecast_diagnostics --run-id safe_rl_wcdt_v2_featurefix_001 --max-samples 512 --low-seed-count 5
```

`forecast_diagnostics.json` 会写入 `forecast_feature_summary`、`forecast_merge_gap_equals_min_distance_rate` 和 `policy_feature_sensitivity`。其中 `forecast_merge_gap` 统一表示未来目标合流车道可用 gap，不再复用 `forecast_min_distance`。如果 WcDT v2 预测质量通过但 zero/shuffle forecast dims 后动作几乎不变，报告会标记 `forecast_policy_underutilized=true`。

如果 20k PPO 下 WcDT v2 policy 收益仍有限，再跑长训练对照：

```powershell
python -m safe_rl.pipeline.run_full_pipeline --run-id safe_rl_wcdt_v2_longppo_001 --forecast-sources constant_velocity,wcdt_v2 --ppo-timesteps 100000
python -m safe_rl.pipeline.stage5_confirmatory_eval --run-id safe_rl_wcdt_v2_longppo_001 --episodes 50
```

如果长训练后 forecast PPO 变激进、尾部安全不稳定，使用 safety-aware forecast PPO。该配置只作用于 forecast PPO，baseline PPO 仍使用默认 reward：

```powershell
python -m safe_rl.pipeline.run_full_pipeline --run-id safe_rl_wcdt_v2_safetyppo_001 --forecast-sources constant_velocity,wcdt_v2 --forecast-ppo-profile safety --forecast-ppo-timesteps 100000
python -m safe_rl.pipeline.stage5_confirmatory_eval --run-id safe_rl_wcdt_v2_safetyppo_001 --episodes 50
python -m safe_rl.pipeline.forecast_diagnostics --run-id safe_rl_wcdt_v2_safetyppo_001 --max-samples 512 --low-seed-count 5
```

如果 safety-aware PPO 仍没有充分利用 WcDT v2 预测特征，下一步使用 Shield-guided forecast PPO。该 profile 仍不替换训练动作，只把 Risk Module / Shield shadow 判断写入 reward shaping，让 forecast PPO 学会避开会被 Shield 替换的 raw action：

```powershell
python -m safe_rl.pipeline.run_full_pipeline --run-id safe_rl_wcdt_v2_shieldguided_001 --forecast-sources constant_velocity,wcdt_v2 --forecast-ppo-profile shield_guided --forecast-ppo-timesteps 100000
python -m safe_rl.pipeline.stage5_confirmatory_eval --run-id safe_rl_wcdt_v2_shieldguided_001 --episodes 50
python -m safe_rl.pipeline.forecast_diagnostics --run-id safe_rl_wcdt_v2_shieldguided_001 --max-samples 512 --low-seed-count 5
```

`shield_guided_forecast` 会在 forecast PPO 生成配置中绑定 base run 的 `stage2/risk_module.pt`。baseline PPO 不加载该 Risk Module，仍保持默认 reward。

Stage3 默认启用 safety checkpoint selection。这里的 `safety_score` / `checkpoint_selection_score` 只用于在多个 checkpoint 中选择下游使用的 `ppo_model.zip`，不是 PPO 训练 reward。训练 reward 由 `rl.reward_profile` 决定。训练结束后会同时输出：

```text
safe_rl_output/runs/<run_id>/stage3/ppo_model_final.zip
safe_rl_output/runs/<run_id>/stage3/ppo_model_best_safety.zip
safe_rl_output/runs/<run_id>/stage3/ppo_model.zip
safe_rl_output/runs/<run_id>/stage3/stage3_checkpoint_selection_report.json
```

下游 Stage5 仍读取 `ppo_model.zip`，该路径会保存/复制为 `checkpoint_selection_score` 最好的 checkpoint，避免最后一个 checkpoint 过于激进。默认 `stage3.checkpoint_selection_profile: "safety"` 保持安全优先；如果后续发现 WcDT v2 policy 过度保守，可以显式改为 `"safety_efficiency"`，在安全项基础上轻量加入 `completion_time_mean` 和 `ego_speed_mean`，但它不是当前默认主线。

如果需要手动逐阶段运行，命令如下：

```powershell
$RUN_ID = "safe_rl_highway_merge_001"

python scenarios\highway_merge\build_network.py
python -m safe_rl.pipeline.stage1_risk_probe --run-id $RUN_ID
python -m safe_rl.pipeline.stage2_train_prediction_risk --run-id $RUN_ID
python -m safe_rl.pipeline.stage3_train_ppo --run-id $RUN_ID
python -m safe_rl.pipeline.stage4_collect_failures --run-id $RUN_ID
python -m safe_rl.pipeline.stage2_train_prediction_risk --run-id $RUN_ID --config safe_rl\config\advanced\stage2_with_stage4.yaml
python -m safe_rl.pipeline.stage5_paired_eval --run-id $RUN_ID
```

## 命令行进度输出

五个 Stage 都会在命令行输出关键运行信息，包括：

```text
run_id
SUMO config / SUMO binary
输入 checkpoint 或 buffer 路径
输出目录
episode / epoch / seed 进度
关键输出文件路径
```

典型示例：

```text
[stage2] run_id=safe_rl_highway_merge_001
[stage2] input_stage1=...\stage1\risk_probe_buffer.npz
[stage2] transition_count=12345
Stage2 risk epochs:  30%|...
[stage2] risk epoch=3/10 loss=0.421337
[stage2] report=...\stage2\stage2_training_report.json
```

## TensorBoard 查看训练效果

默认配置中 `run.tensorboard=true`，各阶段会写入：

```text
stage1/tensorboard/  # episode reward, collision, near-miss, min distance
stage2/tensorboard/  # risk loss, prediction loss
stage3/tensorboard/  # SB3 PPO reward/loss/value/entropy 等
stage4/tensorboard/  # on-policy reward, intervention/fallback/collision
stage5/tensorboard/  # 各实验组 reward/safety/task/intervention 指标
```

启动 TensorBoard：

```powershell
tensorboard --logdir safe_rl_output\runs\$RUN_ID
```

如果要关闭 TensorBoard：

```yaml
run:
  tensorboard: false
```

## SUMO 回放与可视化

Stage1、Stage4、Stage5 会默认写 replay JSON。它记录 seed、action 序列、shield 是否启用、risk checkpoint 和模型路径，用于重新启动 SUMO 回放同一段闭环过程。

无 GUI 回放：

```powershell
python -m safe_rl.tools.replay_episode --replay safe_rl_output\runs\$RUN_ID\stage1\replay\episode_0000.json
```

使用 SUMO-GUI 可视化回放：

```powershell
python -m safe_rl.tools.replay_episode --replay safe_rl_output\runs\$RUN_ID\stage1\replay\episode_0000.json --gui --delay-ms 200
```

Stage5 实验结果回放示例：

```powershell
python -m safe_rl.tools.replay_episode --replay safe_rl_output\runs\$RUN_ID\stage5\replay\ppo_shield_seed_1.json --gui --delay-ms 200
```

Forecast diagnostics 会自动挑出 `ppo_cv_features` 中 min distance 最低的 seeds，并生成可直接运行的回放脚本：

```text
safe_rl_output/runs/<run_id>/stage5/diagnostics/replay_low_min_distance_ppo_cv_features.ps1
safe_rl_output/runs/<run_id>/stage5/diagnostics/replay_low_min_distance_ppo_wcdt_v2_features.ps1
```

也可以对已有 run 手动补生成 forecast diagnostics：

```powershell
python -m safe_rl.pipeline.forecast_diagnostics --run-id $RUN_ID --max-samples 512 --low-seed-count 5
```

如果之后重新运行 confirmatory，`confirmatory_summary.json` 会读取已有 `forecast_diagnostics.json` 并补充 `forecast_policy_utilization_summary`；如果 diagnostics 尚未生成，该字段会以 `available=false` 保留。

如果机器上 `sumo-gui` 不在 `PATH` 中：

```powershell
python -m safe_rl.tools.replay_episode --replay safe_rl_output\runs\$RUN_ID\stage5\replay\ppo_shield_seed_1.json --sumo-binary "E:\Program Files\sumo-1.22.0\bin\sumo-gui.exe" --delay-ms 200
```

关闭 replay 输出：

```yaml
run:
  replay: false
```

## Stage1 数据分布审计

Stage1 完成后会自动生成：

```text
safe_rl_output/runs/<run_id>/stage1/audit/stage1_data_audit.json
safe_rl_output/runs/<run_id>/stage1/audit/stage1_action_histogram.csv
safe_rl_output/runs/<run_id>/stage1/audit/stage1_action_histogram.png
safe_rl_output/runs/<run_id>/stage1/audit/stage1_reward_distribution.png
safe_rl_output/runs/<run_id>/stage1/audit/stage1_risk_distribution.png
```

审计内容包括：

```text
action histogram
overall risk rate
collision / near-miss / low-TTC / high-DRAC / merge-conflict rate
reward 分位数
risk feature 分位数
每个 episode 的 transition 数
trajectory sample 数
```

如果只想跳过审计：

```yaml
stage1:
  audit_enabled: false
```

## 快速自检命令

不依赖完整训练的基础测试：

```powershell
python -m pytest tests\test_safe_rl_core.py
```

Python 语法编译检查：

```powershell
python -m compileall safe_rl tests
```

最小 SUMO 环境 smoke test：

```powershell
python -c "from safe_rl.utils.config import load_config; from safe_rl.sim.sumo_highway_merge_env import SumoHighwayMergeEnv; cfg=load_config(); cfg.scenario['episode_seconds']=1.0; env=SumoHighwayMergeEnv(cfg, seed=1); obs,_=env.reset(seed=1); obs,r,t,tr,info=env.step(4); env.close(); print(obs.shape, r, t, tr)"
```

## 关键实现说明

- `ego` 已改为匝道车辆，merge success 定义为进入 `main_out` 并超过配置中的 `success_min_x`。
- 当前 `highway_merge.rou.xml` 已使用更高难度交通流：合流目标车道 lane 2 为 `1350 veh/h`，lane 1 为 `1150 veh/h`，lane 0 为 `900 veh/h`，匝道为 `650 veh/h`；地图长度和连接关系不变。
- 离散动作空间共 9 个动作：`lateral_cmd {-1,0,+1} x accel_cmd {-1,0,+1}`。
- PPO observation 默认包含 ego 状态、top-k 周车相对状态、merge 几何；启用 forecast features 后拼接低维预测风险特征。
- Risk Module 同时使用显式物理风险特征和学习型 MLP 风险头。
- Shield V2 默认只在 raw action 风险高于 `activation_risk_threshold=0.90`、候选动作风险至少降低 `replacement_margin=0.15` 且 uncertainty 低于阈值时替换；`allow_fallback=false`，没有明显更安全候选动作时继续执行 raw action。
- `emergency_fallback` 不是普通 fallback 的恢复，而是极端物理危险下的低频 backstop；报告中应同时看 `fallback_rate == 0` 和 `emergency_fallback_rate` 是否低频。

## 原始 WcDT 说明

本仓库原始部分来自 WcDT：World-centric Diffusion Transformer for Traffic Scene Generation。原 Waymo 数据预处理与训练入口仍保留：

```powershell
bash run_main.sh
```

原始 WcDT 论文引用：

```bibtex
@article{yang2024wcdt,
  title={Wcdt: World-centric diffusion transformer for traffic scene generation},
  author={Yang, Chen and He, Yangfan and Tian, Aaron Xuxiang and Chen, Dong and Wang, Jianhui and Shi, Tianyu and Heydarian, Arsalan and Liu, Pei},
  journal={arXiv preprint arXiv:2404.02082},
  year={2024}
}
```
