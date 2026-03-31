# DSSGym

基于 [siemens/powergym](https://github.com/siemens/powergym) 的二次开发项目，面向“配电网运行 + EV 充电调度 + 储能控制”的强化学习研究。

当前代码重点是：
- 将环境控制重心调整为电池/充电站相关动作。
- 引入 EV 到达-离开需求队列与充电站管理。
- 使用 PPO 与规则智能体两条流程进行对比测试。

## 项目亮点

1. 功率控制机制（双重约束）
- 充电功率决策同时考虑充电桩额定容量约束与 BMS 请求功率约束，不再只由单一控制量决定。
- 直观上可理解为：实际可执行充电功率受 `桩容量上限` 和 `BMS请求上限` 共同限制（并结合动作映射与运行状态）。

2. 基于 OpenDSS 的高精度配网潮流仿真
- 环境底层依托 OpenDSS 进行配电网潮流计算，而非简化潮流近似模型。
- 训练与测试过程中的电压、功率、网损等指标均来自时序潮流结果。

## 1. 快速开始

### 1.1 环境准备

PowerShell 示例：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

说明：
- `requirements.txt` 中包含 `python==3.11.8`。如果安装时报 `No matching distribution found for python==...`，请删除该行后重试（Python 版本应在虚拟环境创建时指定）。
- 本项目依赖 `dss_python` 与 `dss_python_backend`，请确保本机可正常加载 OpenDSS 相关动态库并已经安装并可调用 OpenDSS 计算引擎。

### 1.2 目录下直接运行

建议在仓库根目录运行：

```powershell
cd \path\to\dssgym
```

## 2. 训练与测试

### 2.1 PPO 训练（训练后自动测试一次）

```powershell
python ppo_agent.py --env_name 13Bus_cbat --num_steps 1000000 --use_plot false
```

结果会输出到 `results/results_时间戳_环境名_步数_其他说明信息/`。

### 2.2 仅测试已有模型

```powershell
python ppo_agent.py --env_name 13Bus_cbat --model_path results\xxx\model\ppo_model.zip --test_only true --use_plot true
```

也可以使用批处理：

```powershell
.\test.bat --env_name 13Bus_cbat --model_path results\xxx\model\ppo_model.zip --test_only true
```

### 2.3 规则智能体测试

```powershell
python rules_agent.py --env_name 13Bus_cbat --test_only true --use_plot true
```

### 2.4 其它模式（PPO 脚本）

其他模式在二次开发中未专门适配，仅做简单调整，不保证结果正确性。

`ppo_agent.py` 的 `--mode` 支持：
- `ppo`（默认）
- `parallel_ppo`
- `episodic_ppo`
- `dss`

## 3. 环境命名与配置

当前注册环境（`dssgym/env_register.py`）包含：
- `13Bus` / `13Bus_cbat` / `13Bus_soc` / `13Bus_cbat_soc`
- `34Bus` / `34Bus_cbat` / `34Bus_soc` / `34Bus_cbat_soc`
- `123Bus` / `123Bus_cbat` / `123Bus_soc` / `123Bus_cbat_soc`
- `8500Node` / `8500Node_cbat` / `8500Node_soc` / `8500Node_cbat_soc`

命名规则：
- `_cbat`：连续电池动作。
- `_soc`：含 SOC 惩罚项版本。
- `_s<数字>`：负载缩放后缀，例如 `13Bus_cbat_s2.0`。

当前统一覆盖配置（在 `env_register.py` 中对所有环境生效）：
- `max_episode_steps = 96`
- `reg_w = 0.0`
- `cap_w = 0.0`
- `soc_w = 5.0`
- `dis_w = 0.0`
- 额外 EV/充电站相关奖励：`connection_w`、`completion_w`、`energy_w`、`voltage_w`、`tf_capacity_w`

注意：
- 目前 `13Bus` 的充电站参数最完整。
- 其他算例的 `bus_name` 在 `_STATION_INFO` 中仍有占位项，使用前建议先按实际系统补全。
- 基础环境来源于 OpenDSS 自带环境，本项目中的 DSS 文件在其基础上进行了更改。

## 4. EV 需求文件配置优先级

环境会按以下优先级解析 EV 需求 CSV（高到低）：
1. 命令行参数 `--ev_demand_path`
2. 环境变量 `DSSGYM_EV_DEMAND_<SYSTEM_NAME>`
3. 环境变量 `DSSGYM_EV_DEMAND_PATH`
4. `env_register.py` 中 `_EV_INFO` 的默认值

PowerShell 示例：

```powershell
$env:DSSGYM_EV_DEMAND_13BUS = "d:\su196\Documents\Python\ML\dssgym\ev_demand\ev_demand-public_parking-general-250-A95.csv"
python ppo_agent.py --env_name 13Bus_cbat --num_steps 100000
```

## 5. 结果文件说明

训练目录（`results/results_.../`）常见文件：
- `model/ppo_model.zip`
- `rewards_in_training.csv`
- `reward_weights.csv`
- `train_env_settings_info.txt`
- `test_results_时间戳/`（自动测试结果）

测试目录（`test_results_.../` 或 `test_results_rules_.../`）常见文件：
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

## 6. 结果分析

可直接调用 `test_results_analysis.py` 中函数：

```powershell
@'
from test_results_analysis import test_results_analysis
test_results_analysis(r"results\results_xxx\test_results_xxx", 1)
'@ | python -
```

## 7. 项目结构（简要）

- `dssgym/env.py`：核心 Gymnasium 环境与奖励函数。
- `dssgym/env_register.py`：环境注册、系统参数、EV 数据路径解析。
- `ppo_agent.py`：PPO 训练与测试入口。
- `rules_agent.py`：规则智能体基线。
- `test_results_analysis.py`：测试结果统计与绘图。
- `systems/`：各 IEEE 配电网算例与曲线数据。
- `ev_demand/`：EV 到达离开与需求样本。

## 8. 开发计划

Next:
- [ ] 完善并扩展负荷生成类
- [ ] 增加 EV 电池的状态管理
- [ ] 使用状态模式管理 EV 电池的不同状态（等待、充电、完成等）

Todo:
- [ ] 启用全局输出控制，可选择关闭以降低消耗
  - [x] 使用 logging
- [ ] 添加多充电站配置
- [ ] 添加功率模块设置
- [ ] 添加路网耦合设置
- [ ] 添加充电电价影响
- [ ] 添加用户行为模型
- [ ] 光伏添加不确定性、可控性

## 9. 来源与许可

- Upstream: [siemens/powergym](https://github.com/siemens/powergym)
- 本仓库采用双许可证：`PolyForm-Noncommercial-1.0.0` 或商业许可证（见 [LICENSE](./LICENSE)）
- 默认仅授权非商用用途（PolyForm Noncommercial 1.0.0）
- 非商用许可证文本：见 [LICENSES/PolyForm-Noncommercial-1.0.0.txt](./LICENSES/PolyForm-Noncommercial-1.0.0.txt)
- 商业授权说明：见 [LICENSES/COMMERCIAL.txt](./LICENSES/COMMERCIAL.txt)
- 商业使用需另行签署商业许可证
- 本仓库为 upstream MIT 项目的二次开发，MIT 权利不会被撤销（见 [NOTICE](./NOTICE)）
- 由于本仓库难以清晰拆分出“仅 MIT 的独立子集”，若你希望按 MIT 条款使用对应代码，请直接使用上游仓库：<https://github.com/siemens/powergym>
