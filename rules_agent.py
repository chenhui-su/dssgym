# -*- coding: utf-8 -*-
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-DSSGym-Commercial
# Copyright (c) 2025 Su Chenhui
# @File: rules_agent.py
# @Time: 2025/5/3 09:56
# @Author: Gan Mocai
# @Software: PyCharm, VS Code

"""
按固定规则行为的智能体，与环境交互。
解析obs，根据obs按照规则进行动作
"""
import datetime

import numpy as np
import random
import time
import os
import argparse

from matplotlib import pyplot as plt

from ppo_agent import build_runtime_config, make_env, parse_arguments, seeding


class RulesAgent:
    """
    基于规则的智能体，与环境交互
    """
    def __init__(self, env):
        """
        初始化规则智能体

        Args:
            env: 环境对象
        """
        self.env = env
        self.cap_num = env.cap_num
        self.reg_num = env.reg_num
        self.bat_num = env.bat_num
        self.sto_num = env.sto_num
        self.reg_act_num = env.reg_act_num
        self.bat_act_num = env.bat_act_num

        # 获取动作和观察空间信息
        self.obs_dim = env.observation_space.shape[0]
        self.CRB_num = (self.cap_num, self.reg_num, self.bat_num)
        self.CRB_dim = (2, self.reg_act_num, self.bat_act_num)

        print('NumCap, NumReg, NumBat: {}'.format(self.CRB_num))
        print('ObsDim, ActDim: {}, {}'.format(self.obs_dim, sum(self.CRB_num)))

    def parse_observation(self, obs):
        """
        解析观察向量，提取有用信息

        Args:
            obs: 环境返回的观察向量

        Returns:
            解析后的观察信息字典
        """
        # 观察空间包含:
        # 1. 节点电压值 - nnode维
        # 2. 电容状态 - cap_num维
        # 3. 调压器状态 - reg_num维
        # 4. 电池状态(SOC和功率) - bat_num*2维
        # 5. 充电站指标 - 5维
        # 6. 负载信息(如果observe_load=True) - nload维

        index = 0
        parsed_obs = {}

        # 1. 获取节点电压值
        nnode = len(np.hstack(list(self.env.obs['bus_voltages'].values())))
        parsed_obs["voltages"] = obs[index:index + nnode]
        index += nnode

        # 2. 获取电容状态
        parsed_obs["cap_statuses"] = obs[index:index + self.cap_num]
        index += self.cap_num

        # 3. 获取调压器状态
        parsed_obs["reg_statuses"] = obs[index:index + self.reg_num]
        index += self.reg_num

        # 4. 获取电池状态(SOC和功率)
        parsed_obs["battery_soc"] = []
        parsed_obs["battery_power"] = []
        for i in range(self.bat_num):
            parsed_obs["battery_soc"].append(obs[index])
            parsed_obs["battery_power"].append(obs[index + 1])
            index += 2

        # 5. 获取充电站指标
        parsed_obs["ev_connection_rate"] = obs[index]
        parsed_obs["ev_charging_power"] = obs[index + 1]
        parsed_obs["ev_success_rate"] = obs[index + 2]
        parsed_obs["ev_connection_count"] = obs[index + 3]
        parsed_obs["ev_avg_satisfaction"] = obs[index + 4]
        index += 5

        # 6. 获取负载信息(如果有)
        if hasattr(self.env, 'observe_load') and self.env.observe_load:
            nload = len(self.env.obs['load_profile_t'])
            parsed_obs["load_profile"] = obs[index:index + nload]
            index += nload

        # 计算时间指标(日内时刻)
        # 根据环境的时间步长计算一天中的时间
        t = self.env.t % self.env.horizon  # 获取日内时间步
        parsed_obs["time_of_day"] = t / self.env.horizon  # 归一化为0-1之间

        return parsed_obs

    def select_action(self, obs):
        """
        基于规则选择动作

        Args:
            obs: 环境返回的观察向量

        Returns:
            actions: 动作向量
        """
        parsed_obs = self.parse_observation(obs)

        # 初始化动作向量
        actions = []

        # # 电容器动作规则 (0表示断开，1表示闭合)
        # for i in range(self.cap_num):
        #     # 示例规则：当负载高时，闭合电容器
        #     if parsed_obs["load"] > 0.7:
        #         actions.append(1)  # 闭合
        #     else:
        #         actions.append(0)  # 断开

        # # 调压器动作规则 (0到reg_act_num-1)
        # for i in range(self.reg_num):
        #     # 示例规则：根据电压设置调压器位置
        #     voltage = parsed_obs["voltage"][i]
        #     if voltage < 0.95:
        #         actions.append(self.reg_act_num - 1)  # 提高电压
        #     elif voltage > 1.05:
        #         actions.append(0)  # 降低电压
        #     else:
        #         actions.append(self.reg_act_num // 2)  # 保持中间位置

        # 电池动作规则——主要根据变压器容量裕度和当前功率操作
        # 计算裕度
        tf_kVA = self.env.station_transformer_capacity
        ev_kW = self.env.obs['ev_charging_power']
        # 添加 clip 保护防止 arccos 产生 NaN
        ev_pf_clipped = np.clip(self.env.ev_station.EV_PF, -1.0, 1.0)
        ev_kVar = ev_kW * np.tan(np.arccos(ev_pf_clipped))
        storage_kW = sum([bat.actual_power() for name, bat in self.env.circuit.storage_batteries.items() if
                          name in self.env.bat_names[:self.env.sto_num]])
        storage_kVar = sum(
            [bat.actual_power() * np.tan(np.arccos(np.clip(bat.pf, -1.0, 1.0))) for name, bat in self.env.circuit.storage_batteries.items()
             if
             name in self.env.bat_names[:self.env.sto_num]])
        PV_kW = sum(load.feature[1] for key, load in self.env.circuit.loads.items() if
                    "PV" in key and load.bus == self.env.ev_station_bus)  # feature0 kV,1 kW, 2 kVar
        PV_kVar = sum(load.feature[2] for key, load in self.env.circuit.loads.items() if
                      "PV" in key and load.bus == self.env.ev_station_bus)
        total_kW = ev_kW + storage_kW + PV_kW
        total_kVar = ev_kVar + storage_kVar + PV_kVar
        total_kVA = np.sqrt(total_kW ** 2 + total_kVar ** 2)
        remain_kVA = 0.75 * tf_kVA - total_kVA
        # 示例规则：根据容量裕度和一天中的时间决定充放电
        time_of_day = parsed_obs["time_of_day"]

        # 判断动作空间类型并适配
        if hasattr(self.env, "action_space") and hasattr(self.env.action_space, "dtype"):
            is_continuous = np.issubdtype(self.env.action_space.dtype, np.floating)
        else:
            is_continuous = False  # 默认假设为离散动作空间

        for i in range(self.bat_num):
            if is_continuous:
                # 连续动作空间 [-1, 1] 范围
                if remain_kVA < -150:  # 变压器超安全容量多
                    # 强制储能放电以减轻负载
                    if i < self.sto_num:
                        actions.append(1.0)
                    else:
                        actions.append(-0.4)
                elif remain_kVA < 0:  # 轻微
                    if i < self.sto_num:
                    # 放电强度与过载程度成正比
                        discharge_level = min(remain_kVA / -150, 1.0)
                        actions.append(discharge_level)
                    else:
                        actions.append(-0.8)
                else:
                    if i < self.sto_num:
                        actions.append(-0.2)
                    else:
                        actions.append(-1.0)
            else:
                # 离散动作空间 [0, bat_act_num-1]
                mid_point = self.bat_act_num // 2  # 中间点表示不充不放

                if remain_kVA < -150:  # 变压器超安全容量多
                    # 强制储能放电以减轻负载
                    if i < self.sto_num:
                        actions.append(0)  # 最大放电
                    else:
                        # 对应连续空间的-0.4，大约是轻微充电
                        actions.append(int(mid_point * (1 - 0.4)))
                elif remain_kVA < 0:  # 轻微
                    if i < self.sto_num:
                        # 放电强度与过载程度成正比
                        discharge_ratio = min(remain_kVA / -150, 1.0)
                        # 将连续空间的discharge_level映射到离散空间
                        discharge_level = int(mid_point * (1 + discharge_ratio))
                        actions.append(max(discharge_level, 0))
                    else:
                        # 对应连续空间的0.8，更强的充电
                        actions.append(int(mid_point * (1 - 0.8)))
                else:
                    if i < self.sto_num:
                        # 对应连续空间的-0.2，轻微充电
                        actions.append(int(mid_point * (1 - 0.2)))
                    else:
                        # 对应连续空间的1.0，最大充电
                        actions.append(0)

        return np.array(actions)


def test_rules_agent(output_dir=None, args=None, load_profile_idx=0,
                     worker_idx=None, use_plot=False, print_step=False, gen_calculate=False):
    """
    测试规则智能体并保存结果到指定目录

    Args:
        output_dir: 输出目录路径(可选)，不指定时使用当前工作目录
        args: 命令行参数
        load_profile_idx: 负载配置索引
        worker_idx: 进程ID
        use_plot: 是否绘图
        print_step: 是否打印每步信息
        gen_calculate: 是否通过Power获取generator类型的功率

    Returns:
        episode_reward: 评估回合的总奖励
        save_path: 结果保存路径
    """
    # 确定保存路径
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if output_dir:
        # 使用指定的输出目录
        base_dir = output_dir
    else:
        # 如果没指定，使用当前工作目录
        base_dir = os.getcwd()

    # 创建结果目录
    save_path = os.path.join(base_dir, f"test_results_rules_{timestamp}_{args.env_name}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)



    # 创建绘图子目录
    plot_dir = os.path.join(save_path, "plots")
    if use_plot and not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # 获取环境
    env = make_env(args.env_name, worker_idx=worker_idx, runtime_config=build_runtime_config(args))
    env.seed(args.seed + 0 if worker_idx is None else worker_idx)

    # 创建规则智能体
    agent = RulesAgent(env)

    # 记录测试信息
    with open(os.path.join(save_path, "test_info.txt"), "w", encoding='utf-8') as f:
        f.write(f"测试时间: {timestamp}\n")
        f.write(f"环境名称: {args.env_name}\n")
        f.write(f"负载配置索引: {load_profile_idx}\n")
        f.write(f"智能体类型: 规则智能体\n")
        f.write(f"随机种子: {args.seed}\n")

    # 准备数据记录文件
    actions_file = os.path.join(save_path, "actions.csv")
    voltages_file = os.path.join(save_path, "voltages.csv")
    total_powers_file = os.path.join(save_path, "total_powers.csv")
    ev_stats_file = os.path.join(save_path, "ev_stats.csv")
    node_powers_file = os.path.join(save_path, "node_powers.csv")
    rewards_file = os.path.join(save_path, "rewards.csv")
    observations_file = os.path.join(save_path, "observations(next).csv")

    # 初始化CSV文件
    with open(actions_file, 'w', encoding='utf-8') as f:
        header = "step"
        action_dim = env.action_space.shape[0] if hasattr(env.action_space, 'shape') else 1
        for i in range(action_dim):
            header += f",action_{i}"
        f.write(header + "\n")

    with open(voltages_file, 'w', encoding='utf-8') as f:
        header = "step"
        for bus_name in env.all_bus_names:
            header += f",{bus_name}"
        f.write(header + "\n")

    with open(total_powers_file, 'w', encoding='utf-8') as f:
        f.write("step,P,Q,PowerLoss,PowerFactor\n")

    with open(ev_stats_file, 'w', encoding='utf-8') as f:
        f.write("step,连接率,充电功率,总连接数,成功率,平均满足率\n")

    with open(node_powers_file, 'w') as f:
        header = "step"
        for bus_name in env.all_bus_names:
            header += f",{bus_name}_P,{bus_name}_Q"
        f.write(header + "\n")

    with open(rewards_file, 'w', encoding='utf-8') as f:
        header = 'step,total_reward'
        for component in env.reward_func.components:
            header += f",{component}"
        f.write(header + "\n")

    # 初始化observations CSV文件
    with open(observations_file, 'w', encoding='utf-8') as f:
        header = "step"
        obs_dim = env.observation_space.shape[0] if hasattr(env.observation_space, 'shape') else 1
        for i in range(obs_dim):
            header += f",obs_{i}"
        f.write(header + "\n")

    # 开始测试
    try:
        # 尝试使用新版 gymnasium 接口
        obs, info = env.reset(seed=args.seed, options={'load_profile_idx': load_profile_idx})
        use_new_interface = True
    except (TypeError, ValueError):
        try:
            # 尝试使用旧版 gym 接口
            obs = env.reset(seed=args.seed, load_profile_idx=load_profile_idx)
            use_new_interface = False
        except TypeError:
            # 使用最原始接口
            obs = env.reset(load_profile_idx=load_profile_idx)
            use_new_interface = False

    episode_reward = 0.0
    done = False
    terminated = False
    truncated = False

    for i in range(env.horizon):
        # 使用规则智能体选择动作
        action = agent.select_action(obs)

        # 与环境交互
        if use_new_interface:
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        else:
            next_obs, reward, done, info = env.step(action)
            terminated = done
            truncated = False

        episode_reward += reward

        if print_step:
            print(f"进程 {worker_idx}, Step {i}\n"
                  f"Action: {action}, Reward: {reward:.4f}, Done: {done}")

        # 记录智能体输出动作
        with open(actions_file, 'a') as f:
            line = f"{i}"
            if isinstance(action, np.ndarray):
                for act_val in action:
                    line += f",{act_val}"
            else:
                line += f",{action}"
            f.write(line + "\n")

        # 记录奖励及其子项
        with open(rewards_file, 'a') as f:
            reward_components = [info.get(comp, 0) for comp in env.reward_func.components]
            f.write(f"{i},{reward}" + "".join([f",{rc}" for rc in reward_components]) + "\n")

        # 记录各节点电压
        with open(voltages_file, 'a') as f:
            line = f"{i}"
            for bus_name in env.all_bus_names:
                # 直接使用环境的circuit对象获取电压数据
                bus_voltage = env.circuit.bus_voltage(bus_name)
                # 只保留实部值(偶数索引)
                bus_voltage_real = [bus_voltage[j] for j in range(len(bus_voltage)) if j % 2 == 0]
                # 计算平均电压值
                voltage_value = sum(bus_voltage_real) / len(bus_voltage_real) if bus_voltage_real else 0
                line += f",{voltage_value}"
            f.write(line + "\n")

        # 记录所有观察值
        with open(observations_file, 'a') as f:
            line = f"{i}"
            if isinstance(next_obs, np.ndarray):
                for obs_val in next_obs:
                    line += f",{obs_val}"
            else:
                line += f",{next_obs}"
            f.write(line + "\n")

        # 记录系统功率
        with open(total_powers_file, 'a') as f:
            active_power = env.circuit.total_power()[0]
            reactive_power = env.circuit.total_power()[1]
            power_loss = env.circuit.total_loss()[0]
            power_factor = active_power / (
                np.sqrt(active_power ** 2 + reactive_power ** 2)) if active_power != 0 or reactive_power != 0 else 0
            # 修正功率因数符号
            power_factor = abs(power_factor) if (active_power < 0 and reactive_power <= 0) or (
                        active_power > 0 and reactive_power >= 0) else -abs(power_factor)

            f.write(f"{i},{active_power},{reactive_power},{power_loss},{power_factor}\n")

        # 记录EV充电站统计数据
        with open(ev_stats_file, 'a') as f:
            connection_rate = env.obs.get('ev_connection_rate', 0)
            charging_power = env.obs.get('ev_charging_power', 0)
            connected_count = env.obs.get('ev_connected_count', 0)
            success_rate = env.obs.get('ev_success_rate', 0)
            avg_target_achieved = env.obs.get('avg_target_achieved', 0)

            f.write(f"{i},{connection_rate},{charging_power},{connected_count},{success_rate},{avg_target_achieved}\n")

        # 记录各节点的功率
        with open(node_powers_file, 'a') as f:
            line = f"{i}"
            for bus_name in env.all_bus_names:
                try:
                    # 使用OpenDSS API获取连接到该节点的设备功率
                    # 设置总线为活动总线
                    env.circuit.dss.ActiveCircuit.SetActiveBus(bus_name)

                    active_power = 0
                    reactive_power = 0

                    # 获取该总线名称以便于查找连接的设备
                    bus_name_dss = env.circuit.dss.ActiveCircuit.ActiveBus.Name

                    # 获取连接到该总线的负荷
                    if env.circuit.dss.ActiveCircuit.Loads.First != 0:
                        while True:
                            # 使用OpenDSS命令获取负荷连接的总线
                            load_name = env.circuit.dss.ActiveCircuit.Loads.Name
                            env.circuit.dss.Text.Commands(f"? load.{load_name}.bus1")
                            load_bus = env.circuit.dss.Text.Result.split(".")[0]

                            if load_bus.lower() == bus_name_dss.lower():
                                # 设置该负荷为活动元件
                                env.circuit.dss.ActiveCircuit.SetActiveElement(f"Load.{load_name}")
                                # 获取负荷的功率
                                powers = env.circuit.dss.ActiveCircuit.ActiveCktElement.Powers
                                # 累加功率（偶数索引是P，奇数索引是Q）
                                if powers is not None and len(powers) > 0:
                                    for j in range(0, len(powers), 2):
                                        active_power -= powers[j]  # 负载消耗为负
                                        if j + 1 < len(powers):
                                            reactive_power -= powers[j + 1]
                            if env.circuit.dss.ActiveCircuit.Loads.Next == 0:  # 迭完了
                                break

                    # 获取连接到该总线的电源
                    if env.circuit.dss.ActiveCircuit.Generators.First != 0:
                        if gen_calculate:
                            while True:
                                gen_name = env.circuit.dss.ActiveCircuit.Generators.Name
                                env.circuit.dss.Text.Commands(f"? generator.{gen_name}.bus1")
                                gen_bus = env.circuit.dss.Text.Result.split(".")[0]

                                if gen_bus.lower() == bus_name_dss.lower():
                                    env.circuit.dss.ActiveCircuit.SetActiveElement(f"Generator.{gen_name}")
                                    powers = env.circuit.dss.ActiveCircuit.ActiveCktElement.Powers
                                    if powers is not None and len(powers) > 0:
                                        for j in range(0, len(powers), 2):
                                            active_power += powers[j]
                                            if j + 1 < len(powers):
                                                reactive_power += powers[j + 1]

                                if env.circuit.dss.ActiveCircuit.Generators.Next == 0:
                                    break
                        else:
                            while True:
                                gen_name = env.circuit.dss.ActiveCircuit.Generators.Name
                                env.circuit.dss.Text.Commands(f"? generator.{gen_name}.bus1")
                                gen_bus = env.circuit.dss.Text.Result.split(".")[0]

                                if gen_bus.lower() == bus_name_dss.lower():
                                    # 累加功率
                                    active_power += env.circuit.dss.ActiveCircuit.Generators.kW  # 正号表示发电
                                    reactive_power += env.circuit.dss.ActiveCircuit.Generators.kvar
                                if env.circuit.dss.ActiveCircuit.Generators.Next == 0:
                                    break

                    # 获取连接到该总线的电容器的功率
                    if env.circuit.dss.ActiveCircuit.Capacitors.First != 0:
                        while True:
                            cap_name = env.circuit.dss.ActiveCircuit.Capacitors.Name
                            env.circuit.dss.Text.Commands(f"? capacitor.{cap_name}.bus1")
                            cap_bus = env.circuit.dss.Text.Result.split(".")[0]

                            if cap_bus.lower() == bus_name_dss.lower():
                                # 设置该电容器为活动元件
                                env.circuit.dss.ActiveCircuit.SetActiveElement(f"Capacitor.{cap_name}")
                                powers = env.circuit.dss.ActiveCircuit.ActiveCktElement.Powers
                                # 电容器主要提供无功功率补偿
                                if powers is not None and len(powers) > 0:
                                    for j in range(0, len(powers), 2):
                                        active_power += powers[j]  # 通常接近0
                                        if j + 1 < len(powers):
                                            reactive_power += powers[j + 1]  # 正值表示提供无功功率

                            if env.circuit.dss.ActiveCircuit.Capacitors.Next == 0:
                                break

                except Exception as e:
                    print(f"获取节点 {bus_name} 功率时出错: {e}. 已设置为零。")
                    active_power, reactive_power = 0, 0

                line += f",{active_power},{reactive_power}"
            f.write(line + "\n")

        # 绘制静态图
        if use_plot:
            fig, _ = env.plot_graph()
            fig.tight_layout(pad=0.1)
            fig.savefig(os.path.join(plot_dir, f"node_voltage_{i:04d}.png"))
            plt.close()

        # 更新状态，准备下一步
        obs = next_obs

        # 如果环境结束，跳出循环
        if done:
            break

    # 导出充电站调度表
    try:
        env.ev_station.export_schedule(output_path=os.path.join(save_path, "schedule.csv"))
        env.ev_station.export_storage_statuses(output_path=os.path.join(save_path, "storage_schedule.csv"))
    except Exception as e:
        print(f"导出充电站调度表失败: {e}。")

    # # 生成系统布局图和动画
    # if use_plot:
    #     try:
    #         # 系统布局图
    #         fig, _ = env.plot_graph(show_voltages=False)
    #         fig.tight_layout(pad=0.1)
    #         fig.savefig(os.path.join(plot_dir, "system_layout.pdf"))
    #         plt.close()
    #
    #         # 生成动画
    #         images = []
    #         filenames = sorted(glob.glob(os.path.join(plot_dir, "node_voltage_*.png")))
    #         for filename in filenames:
    #             images.append(imageio.imread(filename))
    #
    #         if images:  # 确保有图像才生成GIF
    #             imageio.mimsave(
    #                 os.path.join(plot_dir, 'node_voltage.gif'),
    #                 images,
    #                 fps=2,
    #                 loop=0,
    #                 duration=500
    #             )
    #             print(f"已生成GIF动画，包含{len(images)}步")
    #     except Exception as e:
    #         print(f"生成图像或动画失败: {e}")

    # 保存总结数据
    with open(os.path.join(save_path, "summary.txt"), "w", encoding='utf-8') as f:
        f.write(f"总奖励: {episode_reward:.4f}\n")
        f.write(f"步数: {env.horizon}\n")
        # 添加电压违规率统计
        try:
            voltage_data = np.loadtxt(voltages_file, delimiter=',', skiprows=1)
            if voltage_data.shape[0] > 0:
                voltage_values = voltage_data[:, 1:]  # 第一列是步数
                under_voltage_rate = np.mean(voltage_values < 0.95) * 100
                over_voltage_rate = np.mean(voltage_values > 1.05) * 100
                f.write(f"电压低于0.95标幺值比例: {under_voltage_rate:.2f}%\n")
                f.write(f"电压高于1.05标幺值比例: {over_voltage_rate:.2f}%\n")
        except Exception as e:
            print(f"计算电压违规率失败: {e}")

    print(f"测试完成 - 总奖励: {episode_reward:.4f}")
    print(f"测试结果保存在: {save_path}")

    return episode_reward, save_path


def run_rules_agent(args):
    """
    运行基于规则的智能体

    Args:
        args: 命令行参数
    """
    # 检查是否要运行测试模式
    if args.test_only:
        # 运行测试并返回
        test_rules_agent(args=args,
                         load_profile_idx=args.load_profile_idx,
                         use_plot=args.use_plot)
        return

    # 获取环境
    env = make_env(args.env_name, runtime_config=build_runtime_config(args))
    if hasattr(env, "seed"):
        env.seed(args.seed)

    # 创建规则智能体
    agent = RulesAgent(env)

    # 所有可用的负载配置
    profiles = list(range(getattr(env, "num_profiles", 1)))

    # 创建结果目录
    if not os.path.exists("result"):
        os.makedirs("result")

    # 输出文件
    fout = open(f"result/{args.env_name}_rules_{args.seed}.csv", 'w')
    fout.write("episode,profile,reward\n")

    # 训练循环
    total_num_steps = 0  # 追踪所有回合的累计步数
    rewards = []

    for i_episode in range(1, args.num_episodes + 1):
        episode_reward = 0
        episode_steps = 0
        done = False
        load_profile_idx = random.choice(profiles)

        # 检测环境接口版本并适配
        try:
            # 新版 gymnasium 接口
            obs, info = env.reset(seed=args.seed, options={'load_profile_idx': load_profile_idx})
            use_new_interface = True
        except (TypeError, ValueError):
            try:
                # 旧版 gym 接口
                obs = env.reset(seed=args.seed, load_profile_idx=load_profile_idx)
                use_new_interface = False
            except TypeError:
                # 最原始接口
                obs = env.reset(load_profile_idx=load_profile_idx)
                use_new_interface = False

        while not done:
            # 基于规则选择动作
            action = agent.select_action(obs)

            # 与环境交互
            if use_new_interface:
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            else:
                next_obs, reward, done, info = env.step(action)

            episode_steps += 1
            total_num_steps += 1
            episode_reward += reward
            obs = next_obs

        # 记录结果
        rewards.append(episode_reward)
        fout.write(f"{i_episode},{load_profile_idx},{episode_reward:.2f}\n")
        fout.flush()

        print(f"episode: {i_episode}, profile: {load_profile_idx}, "
              f"steps: {episode_steps}, reward: {episode_reward:.2f}")

        if i_episode % 10 == 0:
            avg_reward = sum(rewards[-10:]) / min(10, len(rewards))
            print(f"最近10回合平均奖励: {avg_reward:.2f}")
            print("-" * 60)

        if total_num_steps >= args.num_steps:
            break

    fout.close()
    print(f"规则智能体评估完成。结果已保存到 result/{args.env_name}_rules_{args.seed}.csv")

    # 如果需要，运行最终的测试
    if args.test_after_run:
        print("正在进行最终测试...")
        test_rules_agent(output_dir="result", args=args,
                         load_profile_idx=args.load_profile_idx,
                         use_plot=args.use_plot)


if __name__ == '__main__':
    args = parse_arguments()
    seeding(args.seed)

    # 添加规则智能体特有的参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_episodes', type=int, default=50, help='评估的回合数')
    parser.add_argument('--test_only', action='store_true', help='只运行测试')
    parser.add_argument('--test_after_run', action='store_true', help='训练后运行测试')
    parser.add_argument('--load_profile_idx', type=int, default=0, help='测试时使用的负载配置索引')
    parser.add_argument('--use_plot', action='store_true', help='是否生成图表')

    # 合并args
    rules_args, _ = parser.parse_known_args()
    for key, value in vars(rules_args).items():
        setattr(args, key, value)

    # 确保导入需要的库
    if args.test_only or args.test_after_run or args.use_plot:
        import datetime
        import glob

        try:
            import matplotlib.pyplot as plt
            import imageio
        except ImportError:
            print("警告: 缺少绘图或生成动画所需的库。请安装 matplotlib 和 imageio。")
            args.use_plot = False

    start_time = time.time()
    run_rules_agent(args)
    stop_time = time.time()
    execution_time = stop_time - start_time
    print(f"运行时间: {execution_time:.2f}秒")
