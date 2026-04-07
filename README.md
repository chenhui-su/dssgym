# DSSGym

本仓库是基于 [siemens/powergym](https://github.com/siemens/powergym) 的二次开发项目，当前面向的研究为：
在配电网潮流约束下，联合优化 EV 充电调度与储能控制策略，并比较学习策略与规则策略表现。

## 1. 项目定位

相较 upstream `powergym`，本项目当前的主要改动为：
- 将控制重点从传统配网控制组件调整为电池/充电站。
- 引入 EV 到达-离开过程、等待队列与连接点管理机制。
- 支持 PPO 智能体与规则智能体的并行实验流程。

## 2. 技术特征

1. 功率控制机制（双重约束）
   EV 实际充电功率由双侧共同约束：充电容量与 BMS 请求。

2. 基于 OpenDSS 的高精度配网潮流仿真  
   电压、功率、网损等核心指标由 OpenDSS 时序计算结果给出，不依赖简化潮流近似。

3. 面向实验复现的输出组织  
   训练与测试结果按时间戳归档，支持后处理统计与图形化分析。

## 3. 环境准备

建议使用仓库提供的 conda 环境定义：

```powershell
conda env create -f environment.yml
conda activate dssgym-py312
```

说明：
- 推荐 Python 版本：`3.12`。
- 依赖安装策略为 `conda + pip` 混合模式（见 [environment.yml](./environment.yml) 与 [requirements-pip.txt](./requirements-pip.txt)）。
- 纯 pip 安装可参考 [requirements.txt](./requirements.txt)，但不作为默认路径。
- 运行前请确认本机 OpenDSS 相关动态库可被 `dss-python` 正常加载。

## 4. 运行入口

建议在仓库根目录执行命令：

```powershell
Set-Location D:\path\to\dssgym
```

### 4.1 PPO 训练（默认主路径）

```powershell
python ppo_agent.py --env_name 13Bus_cbat --num_steps 1000000 --use_plot false
```

### 4.2 仅测试已有模型

```powershell
python ppo_agent.py --env_name 13Bus_cbat --model_path results\xxx\model\ppo_model.zip --test_only true --use_plot true
```

### 4.3 规则智能体测试

```powershell
python rules_agent.py --env_name 13Bus_cbat --test_only true --use_plot true
```

### 4.4 `--mode` 说明（[ppo_agent.py](./ppo_agent.py)）

- `ppo`：默认训练路径。
- `parallel_ppo`：并行训练路径（兼容保留）。
- `episodic_ppo`：回合式训练路径（兼容保留）。
- `dss`：legacy 模式，已软废弃；默认不执行，需显式传入 `--allow_legacy_dss true`。

备注：除 `ppo` 主路径外，其它模式未针对当前电池/充电站链路进行充分验证。

## 5. 环境命名与配置

当前注册环境（见 [dssgym/env_register.py](./dssgym/env_register.py)）：
- `13Bus` / `13Bus_cbat` / `13Bus_soc` / `13Bus_cbat_soc`
- `34Bus` / `34Bus_cbat` / `34Bus_soc` / `34Bus_cbat_soc`
- `123Bus` / `123Bus_cbat` / `123Bus_soc` / `123Bus_cbat_soc`
- `8500Node` / `8500Node_cbat` / `8500Node_soc` / `8500Node_cbat_soc`

命名规则：
- `_cbat`：连续电池动作空间。
- `_soc`：含 SOC 惩罚项。
- `_s<scale>`：负载缩放后缀，例如 `13Bus_cbat_s2.0`。

项目当前对注册环境施加统一覆盖参数（详见 [dssgym/env_register.py](./dssgym/env_register.py)）：
- `max_episode_steps = 96`
- `reg_w = 0.0`
- `cap_w = 0.0`
- `soc_w = 5.0`
- `dis_w = 0.0`
- 附加 EV/充电站奖励权重：`connection_w`、`completion_w`、`energy_w`、`voltage_w`、`tf_capacity_w`

注意：
- 目前 `13Bus` 的充电站参数最完整。
- 其他算例的 `bus_name` 在 `_STATION_INFO` 中仍有占位项，使用前建议按实际系统补全。
- 基础环境来源于 OpenDSS 自带环境，本项目中的 DSS 文件在其基础上进行了更改。

## 6. EV 需求文件解析优先级

环境按以下顺序解析 EV 需求 CSV（高优先级覆盖低优先级）：
1. 命令行参数 `--ev_demand_path`
2. 环境变量 `DSSGYM_EV_DEMAND_<SYSTEM_NAME>`
3. 环境变量 `DSSGYM_EV_DEMAND_PATH`
4. `_EV_INFO` 默认值（[dssgym/env_register.py](./dssgym/env_register.py)）

示例：

```powershell
$env:DSSGYM_EV_DEMAND_13BUS = "D:\path\to\dssgym\ev_demand\ev_demand-public_parking-general-250-A95.csv"
python ppo_agent.py --env_name 13Bus_cbat --num_steps 100000
```

## 7. 结果产物

训练目录（`results/results_.../`）常见文件：
- `model/ppo_model.zip`
- `rewards_in_training.csv`
- `reward_weights.csv`
- `train_env_settings_info.txt`
- `test_results_时间戳/`

测试目录（`test_results_.../` / `test_results_rules_.../`）常见文件：
- `summary.txt`
- `rewards.csv`
- `actions.csv`
- `voltages.csv`
- `total_powers.csv`
- `node_powers.csv`
- `ev_stats.csv`
- `schedule.csv`
- `storage_schedule.csv`
- `plots/`（启用绘图时）

## 8. 分析脚本

可通过 [test_results_analysis.py](./test_results_analysis.py) 进行结果统计与可视化：

```powershell
@'
from test_results_analysis import test_results_analysis
test_results_analysis(r"results\results_xxx\test_results_xxx", 1)
'@ | python -
```

## 9. 目录概览

- [dssgym/env.py](./dssgym/env.py)：核心 Gymnasium 环境与奖励定义。
- [dssgym/env_register.py](./dssgym/env_register.py)：环境注册、系统参数、EV 路径解析。
- [ppo_agent.py](./ppo_agent.py)：PPO 训练/测试入口。
- [rules_agent.py](./rules_agent.py)：规则策略入口。
- [test_results_analysis.py](./test_results_analysis.py)：测试结果后处理。
- [systems/](./systems/)：DSS 算例及曲线资源。
- [ev_demand/](./ev_demand/)：EV 需求队列样本与生成逻辑。

## 10. 开发计划

### 10.1 当前进行中（Doing）

- [ ] 利用AI辅助编程，针对 [AI 审计报告](./AUDIT_REPORT.md) 进行修复。

### 10.2 近期计划（Next）

- [ ] 完善并扩展负荷生成类，提升可配置性与可复现性。
- [ ] 增加 EV 电池状态管理，强化运行过程状态可观测性。
- [ ] 使用状态模式管理 EV 电池的不同状态（等待、充电、完成等）。

### 10.3 中期待办（Todo）

- [ ] 启用全局输出控制，可选择关闭以降低消耗
  - [x] 使用 logging
- [ ] 添加多充电站配置
- [ ] 添加功率模块设置
- [ ] 添加路网耦合设置
- [ ] 添加充电电价影响
- [ ] 添加用户行为模型
- [ ] 光伏添加不确定性、可控性

## 11. 许可与来源说明

- Upstream: [siemens/powergym](https://github.com/siemens/powergym)
- 本仓库采用双许可证：`PolyForm-Noncommercial-1.0.0` 或商业许可证（见 [LICENSE](./LICENSE)）
- 默认仅授权非商用用途（PolyForm Noncommercial 1.0.0）
- 非商用许可证文本：见 [LICENSES/PolyForm-Noncommercial-1.0.0.txt](./LICENSES/PolyForm-Noncommercial-1.0.0.txt)
- 商业授权说明：见 [LICENSES/COMMERCIAL.txt](./LICENSES/COMMERCIAL.txt)
- 商业使用需另行签署商业许可证
- 本仓库为 upstream MIT 项目的二次开发，MIT 权利不会被撤销（见 [NOTICE](./NOTICE)）
- 由于本仓库难以清晰拆分出“仅 MIT 的独立子集”，若你希望按 MIT 条款使用对应代码，请直接使用上游仓库：<https://github.com/siemens/powergym>
