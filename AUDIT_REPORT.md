# DSSGym 代码审计报告

**项目概述：** 基于 Siemens/powergym 二次开发的配电网 + EV 充电调度强化学习环境  
**审计范围：** `dssgym/`、`ppo_agent.py`、`rules_agent.py`、`test_results_analysis.py`、`reward_curve.py`、`ev_demand/`、`systems/convert_freq.py` 共 ~3800 行 Python 代码  
**审计日期：** 2026-04-06
**审计者**： glm-5-turbo in CodeX

---

## 一、严重问题（Critical）

### C1. ~~`ev_model_v00.py:84` — SOC 单位不一致~~ [已确认无影响]

> **开发者回应：** `ev_model_v00.py` 中的 `set_soc(self.current_soc * 100)` 是 `ev_bms_v00` 内部的用法，该文件（`ev_model_v00.py` + `ev_bms_v00.py`）**已被弃用或者尚未完成实现并启用**，当前项目实际使用的是 `ev_bms_v01.py` 版本。在其他调用路径中不存在 `*100` 的转换，仿真运行正常，不存在单位不一致问题。

~~`current_soc` 是 0-1 范围，乘以 100 传给 BMS。但 `ev_bms_v00.py` 和 `ev_bms_v01.py` 的 `set_soc()` 验证 `0 <= new_soc <= 1`，对传入的 80 等值会返回 `False` 并**静默拒绝更新**。~~

**状态：** 误报。涉及代码属于已弃用的 v00 版本，不影响当前运行。  
**建议：** 可考虑清理已弃用的 `ev_model_v00.py` / `ev_bms_v00.py` 文件（见 L3）。

### C2. ~~`circuit.py` — `BatteryStationManager.check_departures()` 除零风险~~ [实际不可达]

> **开发者回应：** 当 `target_soc == initial_soc` 时，EV 已处于目标 SOC，**不会来充电**，因此 `check_departures()` 的这段计算逻辑根本不会被触发。该条件在业务逻辑上是不可达的。

```python
target_ratio = min(1.0, (final_soc - initial_soc) / (target_soc - initial_soc))
```

~~当 `target_soc == initial_soc` 时（EV 到达时已处于目标 SOC），分母为零。~~

**状态：** 业务逻辑上不可达，风险极低。若后续需要防御性编程可考虑加保护。  
**建议：** 可作为防御性编程的优化点，添加前置判断以增强鲁棒性。

### C3. `ppo_agent.py:240-408` — 训练/测试循环中每步打开 7+ 个 CSV 文件（原 Critical → Medium）

> **开发者回应：** 确认这是一个优化点。每次 `step` 调用都执行 `open(..., 'a')` 写入 actions、voltages 等文件，然后关闭。实际运行中未出现过文件句柄泄漏问题，但 I/O 次数确实偏多，值得优化。

在 `horizon=96` 步的 episode 中累计执行 672+ 次文件 I/O。性能方面有优化空间。

**影响：** 大量 I/O 对训练速度有一定影响。  
**建议：** 在 episode 开始时一次性打开所有文件，循环结束后统一关闭；或使用内存缓冲区批量写入。

### C4. ~~`rules_agent.py:128-178` — `select_action()` 缺少电容/调压器动作维度~~ [刻意设计]

> **开发者回应：** `cap` 和 `reg` 代码块被**特意注释掉**，因为当前场景不需要调整电容/调压器，其他地方已经处理了这些维度（`cap_num = reg_num = 0`）。如果取消注释反而会导致错误。这是有意为之的设计。

~~`actions = []` 仅追加电池动作，注释掉的 `cap` 和 `reg` 代码块未执行。~~

**状态：** 刻意设计，当前场景下 `cap_num = reg_num = 0`，不需要这些动作维度。  
**建议：** 无需修改。可考虑添加注释说明为何保留注释掉的代码。

### C5. ~~`env.py:~793` — `dss_step()` 中 `soc_errs[self.sto_num]` 索引越界~~ [已废弃方法]

> **开发者回应：** 该方法的场景是控制调压器（regulator），在实现时考虑到充电站运营商无法控制调压器，实际已**移除了该功能的调用**。开发者未使用过此方法，保留的标记/代码属于历史遗留。

~~`dss_step()` 中 `soc_errs` 始终为空列表（该路径不执行电池控制），但代码仍按 `self.sto_num` 索引取值，必定抛出 `IndexError`。~~

**状态：** 该方法已不再被调用，属于废弃代码。保留标记可接受，但应附上说明。  
**建议：** 在代码中添加注释说明该方法已废弃及废弃原因；长期可考虑清理。

### C6. `test_results_analysis.py:14` — 未使用的 sympy 导入（原 Critical → Low）

> **开发者回应：** 确认该导入未使用。可以移除该行。同时项目中其他部分也没有使用 sympy，甚至可以考虑移除 sympy 依赖。

```python
from sympy.matrices.expressions.kronecker import rules
```

该导入从未被使用，变量名 `rules` 可能遮蔽其他有意义的变量。

**状态：** 确认无用导入，可清理。  
**建议：** 删除该行；同时评估是否可以移除 sympy 依赖。

---

## 二、重要问题（High）

### H1. `rules_agent.py:1` — 循环导入设计缺陷

```python
from ppo_agent import build_runtime_config, make_env, parse_arguments, seeding
```

`rules_agent.py` 从 `ppo_agent.py` 导入函数，造成两文件紧密耦合。运行 `rules_agent.py` 会触发 `ppo_agent.py` 的全部顶层代码执行（包括 stable_baselines3 等重型依赖）。

**建议：** 将 `build_runtime_config`、`parse_arguments`、`seeding` 等通用函数提取到 `dssgym/utils.py`。

### H2. `rules_agent.py:704-719` — 双重 argparse 解析冲突

先调用 `ppo_agent.parse_arguments()` 消费 `sys.argv`，再创建第二个 `ArgumentParser` 用 `parse_known_args()` 合并。`--test_only` 在两处定义不一致（`ppo_agent` 用 `lambda` 转换，`rules_agent` 用 `store_true`）。`--use_plot` 同理。

**影响：** 命令行参数行为不可预测。  
**建议：** 统一参数解析入口，或让 `rules_agent` 定义自己的完整 parser。

### H3. `env_register.py` — 模块级副作用覆盖所有环境配置

> 开发者回应：这正是我想要实现的，不过实现方式不够优雅。

```python
for env in _ENV_INFO.keys():
    _ENV_INFO[env]['max_episode_steps'] = 96
    _ENV_INFO[env]['reg_w'] = 0.0
    _ENV_INFO[env]['cap_w'] = 0.0
    _ENV_INFO[env]['soc_w'] = 5.0
    # ...
```

无条件覆盖 `_ENV_INFO` 中所有环境的基础配置值，使原始定义变为死代码。开发者很难意识到原始值已被覆盖。

**建议：** 使用显式的 `_apply_global_defaults(info)` 函数，并注释说明覆盖原因。

### H4. `env_register.py:make_env()` — 文件句柄泄漏

> 开发者回应：二次开发时未正式使用并行训练，但是这未免太防御性编程了。

```python
fin = open(base_file, 'r')
with open(base_file[:-4] + '_' + str(worker_idx) + '.dss', 'w') as fout:
    for line in fin:
        ...
```

`fin` 未使用 `with` 语句，若循环中发生异常则文件句柄泄漏。多个 worker 并行时可能耗尽文件描述符。

**建议：** 将 `fin` 也纳入 `with` 语句管理。

### H5. `env.py:step()` 与 `reset()` 中 `bat_statuses` 构建不一致

> 开发者回应：是个好建议，考虑采纳。

`step()` 使用 `self.circuit.batteries.items()`（动态字典，包含 EV 电池），而 `reset()` 使用 `self.circuit.storage_batteries.items()` + EV 站状态。两个代码路径可能产生不同的键集合，导致包装后的观察数组长度不一致，违反 `observation_space` 契约。

**建议：** 统一 `bat_statuses` 的构建逻辑为一个公共方法。

### H6. `circuit.py` — `storage_batteries` 浅拷贝导致引用共享

```python
self.storage_batteries.update(self.batteries)
```

这是浅拷贝引用。后续向 `self.batteries` 添加 EV 电池时，它们也会出现在 `storage_batteries` 中。`ev_batteries` 字典已创建但从未被使用。

**影响：** 储能电池和 EV 电池在统计中可能被混淆。  
**建议：** 使用 `dict(self.batteries)` 创建独立副本，或在添加 EV 电池时明确排除。

### H7. `ev_bms_v01.py:20-22` — 覆盖内建 `print` 函数

> 开发者回应：这是我在探索输出信息控制的产物，你报告的可能后果，其实也完全不会发生，只会导入 EVBMS

```python
print = lambda *args, **kwargs: None
```

在模块顶部覆盖 `print`。如果其他模块使用 `from ev_bms_v01 import *`，会导入被静默的 `print`，导致调试信息被意外吞掉。

**建议：** 使用 `logging` 模块控制输出，不要覆盖内建函数。

### H8. `rules_agent.py:154` — `np.arccos` 可能产生 NaN

> 开发者回应：有道理，但是古法编程时没必要，现在确实可以修一下。

```python
np.tan(np.arccos(self.env.ev_station.EV_PF))
```

如果 `EV_PF` 因浮点精度略超出 `[-1, 1]`（如 `1.0000001`），`arccos` 返回 `NaN`，传播到所有后续功率计算。

**建议：** 添加 `EV_PF = np.clip(EV_PF, -1.0, 1.0)` 保护。

### H9. `rules_agent.py:183 vs 205` — 储能电池索引边界不一致

> 开发者回应：这确实可能是一个问题，但我当时应该是考虑到了，具体是像什么妥协还是特殊的机制限制，我忘了。

连续动作分支用 `i <= self.sto_num`，离散分支用 `i < self.sto_num`。同一电池在不同动作空间类型下得到不同处理。

**建议：** 统一使用 `<` 或 `<=`。

### H10. `env.py` — `power_loss` 除零风险

> 开发者回应：经典除零风险，很多 AI vibe coding 半天没有实质推进，就是这些实际中不会碰到的问题被不理解场景的 AI 折腾半天。

```python
self.obs['power_loss'] = - self.circuit.total_loss()[0] / self.circuit.total_power()[0]
```

`total_power()[0]` 在无负载场景下可能为零，产生 `inf` 或 `NaN`。此处未使用 `+1e-10` 保护（仅 `info` 字典中有）。

**建议：** 添加分母保护 `total_power()[0] + 1e-10`。

### H11. `ppo_agent.py:365 vs 345/379/400` — `Next()` vs `Next` 不一致

> 开发者回应：这么用是有原因的，不规范是上游的问题，而且确实都是兼容的，不过后续会统一的。

Line 365 调用 `.Next()`（方法调用），其他行使用 `.Next`（属性访问）。如果是 COM dispatch wrapper，一种形式会崩溃。

**建议：** 确认 OpenDSS COM 接口约定，统一调用方式。

### H12. `ppo_agent.py:237` — 仅适配 gymnasium 新 API

> 开发者回应：本来就不需要兼容旧版，就是之前的那个 powergym 代码写的太古风了（不仅仅是这个gym的接口），我才适配了新版接口。

`obs, info = env.reset(...)` 假定新版 gymnasium 接口。`rules_agent.py` 中有新旧接口兼容逻辑，但此处没有。

**建议：** 添加旧版 gym 接口的 fallback。

---

## 三、中等问题（Medium）

| 编号 | 位置 | 问题描述 |
|------|------|----------|
| M1 | `ppo_agent.py:84-92` | `seeding()` 不包含 PyTorch 种子，但 `seeding_all()` 有；主程序只调用 `seeding()`，影响可复现性 |
| M2 | `env.py:~1060` | `reset_obs_space()` 调用 `reset()` 建立观测维度，副作用重置电路状态，增加初始化开销 |
| M3 | `env.py:step()` | `info` 字典同时承载奖励子项和环境调试信息，建议拆分为 `info['reward_breakdown']` 和 `info['metrics']` |
| M4 | `env.py` | `plotting()` 依赖 `os.getcwd()` 创建 `plots/`，CWD 变化时路径不一致 |
| M5 | `rules_agent.py:154-160` | 直接访问 `self.env.obs`、`self.env.bat_names`、`self.env.sto_num` 等内部属性，违反封装 |
| M6 | `env_register.py:350` | `re.match` 正则 `+?` 使得 `13Bus_s` 能匹配但 `float('')` 会抛 `ValueError` |
| M7 | `ev_demand/ev_demand.py` | `generate_power_profile()` 当 `duration <= 2` 时索引重叠导致功率曲线错误 |
| M8 | `ev_demand/ev_demand.py:__main__` | `dist_location, dist_day_type = os.path.basename(base_path).split('-')` 假设恰好一个 `-` |
| M9 | `ev_demand/ev_demand.py` | `np.random.seed(22)` 硬编码，不可配置 |
| M10 | `reward_curve.py:83-84` | 分位数计算错误：`quantile(0.000005)` 不是 5% 分位数，`quantile(1)` 不是 95% 分位数 |
| M11 | `reward_curve.py:122` | `plt.show()` 在无头环境下会阻塞或报错 |
| M12 | `circuit.py:200` | `Battery.step_after_solve` 中 `round(kwh)` 对小时间步长造成精度损失 |
| M13 | `circuit.py:105` | `Circuits.__init__` 中 DSS COM 命令无 try/except，坏电路文件会导致未处理异常 |
| M14 | `circuit.py:850-860` | `set_all_batteries_after_solve()` 遍历动态字典，若迭代中 EV 断开会抛 `RuntimeError` |
| M15 | `loadprofile.py:160` | `sinterval=60*60*24//self.steps` 对非整除步数会丢失秒数 |
| M16 | `loadprofile.py:280` | `os.listdir()` 返回所有文件包括非 CSV，可能导致 `pd.read_csv` 失败 |
| M17 | `reward_monitor_callback.py:35` | 直接访问 SB3 内部 `training_env.buf_rews`，是私有 API，跨版本可能失效 |
| M18 | `reward_monitor_callback.py:20` | `if reward not in self.rewards` 对列表做 `in` 操作为 O(n)，长期训练退化为 O(n²) |
| M19 | `convert_freq.py:process_xmatrix` | 字符串替换使用 `find()` 定位，对重复数字会替换到错误位置 |
| M20 | `convert_freq.py:64` | 帮助信息写的是 `convert_linecodes.py`，实际文件名是 `convert_freq.py` |
| M21 | `ev_bms_v01.py:130` | `set_soc()` 在 `is_charging=True` 时用旧的 `current_charge_power` 重算功率 |
| M22 | `end_projection.py:40` | 当 `sto_num=0` 且 `bat_num>0` 时创建 `MultiDiscrete([0,0,...])` 无效动作空间 |

---

## 四、低风险问题（Low）

| 编号 | 位置 | 问题描述 |
|------|------|----------|
| L1 | `test_results_analysis.py:17` | 未使用的 import：`math`、`re`、`glob` |
| L2 | `ev_demand/ev_demand.py` | 未使用的 import：`ListedColormap` |
| L3 | `ev_bms_v00.py` | 存在但未被引用（`circuit.py` 只导入 v01），属于死代码 |
| L4 | `ppo_agent.py:95-111` | `seeding_all()` 已定义但从未调用 |
| L5 | `ppo_agent.py:27` | `imageio.v2` 在新版 imageio 中可能不存在 |
| L6 | `ppo_agent.py:711` | `run_dss_agent` 文档字符串为空 |
| L7 | `rules_agent.py:728` | `import imageio` 未使用 `.v2`，与 `ppo_agent.py` 不一致 |
| L8 | `env.py:10` | `import math` 未使用（注释说"为向上取整引入"） |
| L9 | `env.py:21` | `gymnasium.utils.seeding` 在新版 gymnasium 中已弃用 |
| L10 | `convert_freq.py:56` | 正则中 `\[` 在 raw string 中是冗余转义 |
| L11 | `test_results_analysis.py` / `reward_curve.py` | 模块级 `plt.rcParams` 修改影响全局 matplotlib 状态 |
| L12 | 多处 | 类型注解不一致：部分文件完整，核心文件几乎没有 |
| L13 | 多处 | 大量 `print()` 调试语句与 `logging` 混用 |

---

## 五、Windows 硬编码路径问题

> 开发者回应：这是当时做本科毕设时用于绘制图片的，确实可以考虑修改，但古法编程时没必要。

以下文件包含 Windows 绝对路径或反斜杠分隔符，在 Linux/macOS 上无法工作：

- `test_results_analysis.py:~310`：`r'\dssgym\systems'` 无效绝对路径
- `test_results_analysis.py` / `reward_curve.py` 注释中大量 `r'D:\LENOVO\Documents\...'` 路径

**建议：** 使用 `Path(__file__).resolve().parent` 构建路径，或依赖环境变量/配置。

---

## 六、代码质量总结

| 维度 | 评分 | 说明 |
|------|------|------|
| **架构设计** | 6/10 | 单文件过大（`circuit.py` ~1300 行、`env.py` ~1200 行），模块边界不够清晰 |
| **代码可读性** | 7/10 | 中文注释详尽，但文件内部组织可以更好 |
| **安全性** | 7/10 | C1（误报，已弃用代码）、C2（业务不可达）、C5（已废弃方法）均确认无实际影响 |
| **可维护性** | 5/10 | 循环导入（H1）、参数解析冲突（H2）、模块级副作用（H3）、浅拷贝引用共享（H6） |
| **测试覆盖** | 2/10 | 无单元测试，仅靠集成测试验证 |
| **文档** | 8/10 | README 非常详尽，代码内注释丰富，但缺乏 API 文档 |

---

## 七、优先修复建议

### ~~立即修复（P0）~~ → 已确认问题较少

> 原 P0 列表中的 C1、C2、C5 经开发者确认均不影响当前运行：
> - **C1**：涉及已弃用的 `ev_bms_v00`，不影响当前代码
> - **C2**：`target_soc == initial_soc` 时 EV 不会来充电，逻辑不可达
> - **C5**：`dss_step()` 方法已废弃，不再被调用
> - **C6**：降级为 Low，可在清理时顺带处理

~~1. **C1**：统一 EV SOC 单位，消除 BMS 与 EV model 之间的单位混淆——这是最影响仿真正确性的问题~~
~~2. **C5**：修复 `dss_step()` 中的 `IndexError`~~
~~3. **C2**：修复 `check_departures()` 中的除零风险~~
~~4. **C6**：删除 `test_results_analysis.py` 中无用的 sympy 导入~~

### 短期修复（P1）
5. **C3**（已降级为 Medium）：优化训练循环中的文件 I/O；**H4**：使用 `with` 语句管理文件操作
6. **H1 + H2**：消除循环导入，统一参数解析入口
7. **H7**：移除 `ev_bms_v01.py` 中对 `print` 的覆盖
8. **H5**：统一 `bat_statuses` 构建逻辑
9. **H6**：修复 `storage_batteries` 浅拷贝问题

### 中期重构（P2）
10. **H3**：将模块级配置覆盖改为显式函数
11. 拆分 `circuit.py`（~1300 行）为多个聚焦模块：`circuit.py`、`battery.py`、`ev_station.py`
12. **M10**：修复 `reward_curve.py` 分位数计算逻辑
13. **H8**：为 `arccos` 调用添加 `np.clip` 保护

### 长期改进（P3）
14. 补充核心计算逻辑的单元测试（BMS 功率曲线、奖励函数、动作映射）
15. 统一使用 `logging` 替代 `print` 调试输出
16. 统一类型注解风格
17. 替换所有硬编码路径为相对路径或配置项
