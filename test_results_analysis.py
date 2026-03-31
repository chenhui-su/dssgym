# -*- coding: utf-8 -*-
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-DSSGym-Commercial
# Copyright (c) 2025 Su Chenhui
# @File: test_results_analysis.py
# @Time: 2025/5/6 15:49
# @Software: PyCharm, VS Code

"""
对测试结果进行分析，包括根据测试结果绘制调度图（参考历史实现），读取测试结果记录并进行简单统计分析。
"""
import datetime
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import math
import re
import glob
from pathlib import Path

from sympy.matrices.expressions.kronecker import rules

# 定义中英文字体映射
plt.rcParams['font.family'] = ['serif', 'sans-serif']
# # 设置英文和数字字体为 Times New Roman
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
# 设置中文字体为宋体
plt.rcParams['font.sans-serif'] = ['SimSun'] + plt.rcParams['font.sans-serif']
# 解决坐标轴负数的负号显示问题
plt.rcParams['axes.unicode_minus'] = False

# 其他字体属性设置
# plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 12

# 如果要对特定元素单独设置字体，可以用以下方式
# 例如：设置标题
# plt.title('标题中文 English 123', fontproperties=fm.FontProperties(fname='SimSun.ttf'), fontname='Times New Roman')


def load_schedule_data(file_path):
    """
    加载充电数据文件

    参数:
        file_path: 充电数据文件的路径

    返回:
        ev_ids: 电动汽车ID列表
        charging_powers: 充电功率数组，形状为(n_evs, n_timesteps)
    """
    # 检查第一行是否为表头
    with open(file_path, 'r') as f:
        first_line = f.readline().strip()
        has_header = 'step_0' in first_line

    if has_header:
        df = pd.read_csv(file_path)
        # 获取第一列的电动汽车ID
        ev_ids = df.iloc[:, 0].tolist()
        # 获取除第一列外的所有数据列
        charging_powers = df.iloc[:, 1:].astype(float).values
    else:
        df = pd.read_csv(file_path, header=None)
        # 提取EV ID和充电功率
        ev_ids = df[0].tolist()
        charging_powers = df.iloc[:, 1:].astype(float).values
    return ev_ids, charging_powers


def load_ev_stats_data(file_path):
    """
    加载电动汽车统计数据文件最后一行数据

    Args:
        file_path (str): 统计数据文件路径

    Returns:
        dict: 包含电动汽车统计信息的字典
    """
    if not os.path.exists(file_path):
        return None

    df = pd.read_csv(file_path, header=None)
    if len(df) == 0:
        return None

    # 获取最后一行数据
    last_row = df.iloc[-1]

    # 解析数据并存入字典
    stats_dict = {
        'step': int(last_row[0]),
        'connection_rate': float(last_row[1]),
        'charging_power': float(last_row[2]),
        'total_connections': int(last_row[3]),
        'success_rate': float(last_row[4]),
        'avg_satisfaction': float(last_row[5])
    }

    return stats_dict


def load_storage_data(file_path):
    """
    加载储能放电数据文件

    参数:
        file_path: 放电数据文件的路径

    返回:
        ev_ids: 储能ID列表
        charging_powers: 放电功率数组，形状为(n_evs, n_timesteps)
    """
    # 检查第一行是否为表头
    with open(file_path, 'r') as f:
        first_line = f.readline().strip()
        has_header = 'step_0' in first_line

    if has_header:
        df = pd.read_csv(file_path)
        # 获取第一列的储能ID
        bs_ids = df.iloc[:, 0].tolist()
        # 获取除第一列外的所有数据列
        discharging_powers = df.iloc[:, 1:].astype(float).values
    else:
        df = pd.read_csv(file_path, header=None)
        # 提取储能 ID和充电功率
        bs_ids = df[0].tolist()
        discharging_powers = df.iloc[:, 1:].astype(float).values
    return bs_ids, discharging_powers


def load_summary_data(file_path):
    """
    加载总结数据

    Args:
        file_path (str): 总结文件路径

    Returns:
        dict: 总结数据
    """
    summary = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            if '总奖励:' in line:
                summary['total_reward'] = float(line.split(':')[1].strip())
            elif '步数:' in line:
                summary['steps'] = int(line.split(':')[1].strip())
            elif '电压低于0.95标幺值比例:' in line:
                summary['undervoltage_ratio'] = float(line.split(':')[1].strip().replace('%', ''))
            elif '电压高于1.05标幺值比例:' in line:
                summary['overvoltage_ratio'] = float(line.split(':')[1].strip().replace('%', ''))
    return summary


def load_rewards_data(file_path):
    """
    加载奖励数据文件并解析为字典

    Args:
        file_path (str): 奖励数据文件路径

    Returns:
        dict: 包含奖励数据的字典，可通过rewards['key'][index]方式查询值
    """
    if not os.path.exists(file_path):
        print(f"警告: 没有找到{file_path}文件")
        return None

    rewards = {}
    df = pd.read_csv(file_path)

    # 将列名作为字典的键
    for column in df.columns:
        rewards[column] = df[column].values

    return rewards


def analyze_schedule_and_stats(schedule, ev_stats):
    """
    分析充电调度，提取性能统计数据。

    Args:
        schedule (ndarray): 充电功率数据
        ev_stats (dict): 统计信息

    Returns:
        dict: 分析结果
    """
    # 服务车辆数（总连接数）：有充电功率>0的车辆
    served_vehicles = np.sum(np.any(schedule > 0, axis=1))

    # 计算每个车辆的充电总量 (kWh)
    # 假设每个时间段为15分钟(0.25小时)
    ev_total_energy = np.sum(schedule, axis=1) * 0.25

    # 完全满足的车辆数
    success_vehicles = ev_stats['success_rate'] * ev_stats['total_connections']

    # 平均满足率
    average_satisfaction = ev_stats['avg_satisfaction']

    # 总充电电量 (kWh)
    total_charging_energy = np.sum(ev_total_energy)

    # 平均单车充电量 (kWh)
    avg_ev_energy = total_charging_energy / served_vehicles if served_vehicles > 0 else 0

    # 最大总充电功率 (kW)：所有车辆在某一时刻的总充电功率最大值
    max_charging_power = np.max(np.sum(schedule, axis=0))

    return {
        'served_vehicles': served_vehicles,
        'success_vehicles': success_vehicles,
        'average_satisfaction': average_satisfaction,
        'total_charging_energy': total_charging_energy,
        'avg_ev_energy': avg_ev_energy,
        'max_charging_power': max_charging_power
    }


def generate_summary_stats(results_dir):
    """
    生成结果目录的统计数据

    Args:
        results_dir (str): 结果目录路径

    Returns:
        dict: 统计数据
    """
    # 充电功率文件
    schedule_file = os.path.join(results_dir, 'schedule.csv')

    # 充电统计文件
    ev_stats_file = os.path.join(results_dir, 'ev_stats.csv')

    # 储能放电文件
    storage_file = os.path.join(results_dir, 'storage_schedule.csv')

    # summary 文件
    summary_file = os.path.join(results_dir, 'summary.txt')

    # 奖励记录文件
    rewards_file = os.path.join(results_dir, 'rewards.csv')

    # 加载充电数据
    if os.path.exists(schedule_file):
        ev_ids, charging_powers = load_schedule_data(schedule_file)
    else:
        raise FileNotFoundError("没找到schedule.csv！")

    # 加载统计数据
    if os.path.exists(ev_stats_file):
        ev_stats = load_ev_stats_data(ev_stats_file)
    else:
        raise FileNotFoundError("没找到ev_stats.csv！")

    # 加载储能数据
    if os.path.exists(storage_file):
        storage = load_storage_data(storage_file)
    else:
        raise FileNotFoundError("没找到storage_schedule.csv！")

    # 加载总结数据
    if os.path.exists(summary_file):
        summary_data = load_summary_data(summary_file)
    else:
        raise FileNotFoundError("没找到summary.txt")

    # 加载奖励数据
    rewards_data = load_rewards_data(rewards_file)

    # 分析充电调度
    charging_analysis = analyze_schedule_and_stats(charging_powers, ev_stats)

    # 计算超专变容量运行时间
    # 假设每检测到总奖励函数负值则增加0.25小时
    overload_hours = 0.0
    for tf in rewards_data['Transformer_reward']:
        if tf <= -10.0/200 * 200:
            overload_hours += 0.25

    # 总PV发电量(kWh)
    # 保持为常数一致就行
    total_pv_energy = 460.17

    # 合并结果
    results = {
        **charging_analysis,
        'steps': summary_data['steps'],
        'overload_hours': overload_hours,
        'total_pv_energy': total_pv_energy,
        'undervoltage_ratio': summary_data['undervoltage_ratio'],
        'overvoltage_ratio': summary_data['overvoltage_ratio']
    }

    return results


def visualize_ev_schedule(charging_powers, ev_ids):
    """
    可视化EV充电调度

    Args:
        charging_powers (ndarray): 充电功率数据
        ev_ids (list): EV ID列表

    Returns:
        Figure: Matplotlib图形对象
    """
    # 设置图表尺寸
    plt.figure(figsize=(14, 10))

    # 自定义颜色映射
    custom_cmap = ListedColormap(['white'] + [plt.cm.YlOrRd(i) for i in np.linspace(0, 1, 100)])

    # 绘制热图
    plt.imshow(charging_powers, aspect='auto', cmap=custom_cmap,
               vmin=0, vmax=120, interpolation='nearest')

    # 添加颜色条
    cbar = plt.colorbar(label='充电功率 (kW)')

    # 设置坐标轴
    plt.ylabel('车辆编号')
    plt.xlabel('时间段')

    # 设置横轴为实际时刻（每个时间步为15分钟）
    T = charging_powers.shape[1]
    tick_positions = np.arange(0, T, 4)
    tick_labels = [f"{j // 4 :02d}:00" if j % 4 == 0 else "" for j in tick_positions]
    plt.xticks(tick_positions, tick_labels, fontsize=8)

    # 设置纵轴标签为EV ID
    if len(ev_ids) <= 20:
        # 提取数字部分作为标签
        formatted_ids = [re.search(r'\d+', ev_id).group() if re.search(r'\d+', ev_id) else ev_id for ev_id in ev_ids]
        plt.yticks(np.arange(len(ev_ids)), formatted_ids)
    else:
        # 如果EV数量过多，只显示部分标签
        step = len(ev_ids) // 10
        formatted_ids = [re.search(r'\d+', ev_ids[i]).group() if re.search(r'\d+', ev_ids[i]) else ev_ids[i]
                        for i in range(0, len(ev_ids), step)]
        plt.yticks(np.arange(0, len(ev_ids), step), formatted_ids)

    # plt.title('电动汽车充电调度')

    return plt.gcf()


def visualize_ev_schedule_revised(charging_powers, ev_ids, storage_data=None, pv_data=None, transformer_capacity=800):
    """
    可视化EV充电调度，包括储能放电功率、光伏功率和视在功率。

    Args:
        charging_powers (ndarray): 充电功率数据
        ev_ids (list): EV ID列表
        storage_data (tuple, optional): 储能数据元组 (bs_ids, discharging_powers)
        pv_data (ndarray, optional): 光伏功率数据
        transformer_capacity (float, optional): 专变容量(kVA)，默认为800kVA
    Returns:
        Figure: Matplotlib图形对象
    """
    # 设置图表尺寸
    width_inches = 22 / 2.54  # 转换为英寸
    height_inches = width_inches * (8 / 12)

    # # 创建共享X轴的两个子图
    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(width_inches, height_inches), sharex=True)

    # 创建共享X轴的两个子图，第一个子图占据2/3的纵向空间
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(width_inches, height_inches), sharex=True,
                                   gridspec_kw={'height_ratios': [2, 1]})  # 设置高度比例为2:1

    # 自定义颜色映射
    custom_cmap = ListedColormap(['white'] + [plt.cm.YlOrRd(i) for i in np.linspace(0, 1, 100)])

    # 在主图中绘制热图
    im = ax1.imshow(charging_powers, aspect='auto', cmap=custom_cmap,
                    vmin=0, vmax=120, interpolation='nearest')

    # 配置第一个子图
    ax1.set_title('电动汽车实际充电功率热图')
    ax1.set_ylabel('车辆编号')

    # 添加颜色条
    cbar_ax = fig.add_axes([0.01, 0.10, 0.01, 0.80])  # [左, 下, 宽, 高]
    cbar = fig.colorbar(im, cax=cbar_ax, label='充电功率 (kW)')

    # 设置纵轴标签为EV ID
    n_evs = len(ev_ids)
    if n_evs <= 20:
        # 提取数字部分作为标签
        formatted_ids = [re.search(r'\d+', ev_id).group() if re.search(r'\d+', ev_id) else ev_id for ev_id in ev_ids]
        ax1.set_yticks(np.arange(len(ev_ids)))
        ax1.set_yticklabels(formatted_ids)
    else:
        # 如果EV数量过多，只显示部分标签
        step = max(1, n_evs // 5)
        positions = np.arange(0, n_evs, step)
        formatted_ids = [re.search(r'\d+', ev_ids[i]).group() if re.search(r'\d+', ev_ids[i]) else ev_ids[i]
                         for i in positions]
        ax1.set_yticks(positions)
        ax1.set_yticklabels(formatted_ids)

    # 第二个子图
    # 设置颜色、线条样式、线宽、透明度
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#ff7f00', '#800080']
    linestyles = ['--', '-', '-', ':', '-']
    linewidths = [1, 1, 1, 1, 1.5]
    alphas = [0.5, 0.9, 0.9, 0.5, 0.7]
    time_steps = np.arange(charging_powers.shape[1])
    # EV充电有功功率曲线
    ev_total_active_power = np.sum(charging_powers, axis=0)
    ax2.plot(time_steps, ev_total_active_power, color=colors[0], linestyle=linestyles[0],
             linewidth=linewidths[0], alpha=alphas[0], label='总汽车充电有功功率')
    
    # 计算EV无功功率
    phi_ev = math.acos(-0.98)
    ev_reactive_power = ev_total_active_power * math.tan(phi_ev)
    
    # 初始化专变内净有功功率和净无功功率
    station_active_power = ev_total_active_power.copy()
    station_reactive_power = ev_reactive_power.copy()
    
    # 绘制储能放电功率（如果有）
    if storage_data is not None:
        bs_ids, discharging_powers = storage_data
        total_discharge = np.sum(discharging_powers, axis=0)
        ax2.plot(time_steps[:len(total_discharge)], total_discharge, color=colors[1], linestyle=linestyles[1],
             linewidth=linewidths[1], alpha=alphas[1], label='储能放电有功功率')
        
        # 更新站内有功功率
        phi_storage = math.acos(0.95)
        storage_reactive_power = total_discharge * math.tan(phi_storage)
        station_active_power -= total_discharge[:len(station_active_power)]  # 储能放电减少净负荷
        station_reactive_power -= storage_reactive_power[:len(station_reactive_power)]
    
    # 绘制光伏功率（如果有）
    if pv_data is not None and len(pv_data) > 0:
        # 确保光伏数据长度与时间步数匹配
        if len(pv_data) >= len(time_steps):
            ax2.plot(time_steps, pv_data[:len(time_steps)], color=colors[2], linestyle=linestyles[2],
             linewidth=linewidths[2], alpha=alphas[2], label='光伏出力有功功率')
            # 更新净有功功率 (光伏减少负荷)
            station_active_power -= pv_data[:len(time_steps)]
        else:
            # 如果光伏数据不够长，则填充零值
            pv_extended = np.concatenate([pv_data, np.zeros(len(time_steps) - len(pv_data))])
            ax2.plot(time_steps, pv_extended, color=colors[2], linestyle=linestyles[2],
             linewidth=linewidths[2], alpha=alphas[2], label='光伏出力有功功率')
            # 更新净有功功率
            station_active_power -= pv_extended
    
    # 绘制整站功率曲线
    ax2.plot(station_active_power, color=colors[3], linestyle=linestyles[3],
             linewidth=linewidths[3], alpha=alphas[3], label='充电站有功功率')
    station_apparent_power = np.sqrt(station_active_power**2 + station_reactive_power**2)
    ax2.plot(time_steps, station_apparent_power, color=colors[4], linestyle=linestyles[4],
             linewidth=linewidths[4], alpha=alphas[4], label='充电站视在功率')
    
    # 添加专变容量限制线
    ax2.axhline(y=transformer_capacity, color='red', linestyle='--', alpha=0.7, linewidth=2, label=f'专变容量限制 ({transformer_capacity}kVA)')

    # 配置第二个子图
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc='upper right', borderaxespad=0, fontsize=8)
    # 设置Y轴范围，确保所有曲线都能显示
    all_powers = np.concatenate([
        ev_total_active_power,
        station_apparent_power,
        [0, transformer_capacity * 1.1]  # 确保限制线可见并有一些余量
    ])
    
    if storage_data is not None:
        all_powers = np.concatenate([all_powers, total_discharge])
    
    if pv_data is not None and len(pv_data) > 0:
        if len(pv_data) >= len(time_steps):
            all_powers = np.concatenate([all_powers, pv_data[:len(time_steps)]])
        else:
            pv_extended = np.concatenate([pv_data, np.zeros(len(time_steps) - len(pv_data))])
            all_powers = np.concatenate([all_powers, pv_extended])
    
    y_min = max(min(0, np.min(all_powers) - 10), -50)  # 下限不低于-50
    y_max = max(np.max(all_powers) + 10, transformer_capacity * 1.1)
    ax2.set_ylim(y_min, y_max)
    ax2.set_ylabel('功率 (kW/kVA)')

    ax2.set_xlim(0, charging_powers.shape[1] - 1)
    # 设置横轴为实际时刻（每个时间步为15分钟）
    T = charging_powers.shape[1]
    tick_positions = np.arange(0, T, 4)
    tick_labels = [f"{j // 4 :02d}:00" if j % 4 == 0 else "" for j in tick_positions]
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels, fontsize=8)
    ax2.set_xlabel('时间 (时:分)')

    # 添加垂直分隔线表示每6小时
    for hour in range(6, 24, 6):
        ax1.axvline(x=hour * 4, color='gray', linestyle='--', alpha=0.5)
        ax2.axvline(x=hour * 4, color='gray', linestyle='--', alpha=0.5)

    # 调整布局，为左侧的颜色条留出空间
    plt.subplots_adjust(right=0.99, left=0.15, bottom=0.08, top=0.95)

    return plt.gcf()


def load_voltage_data(file_path, excluded_nodes=None):
    """
    加载节点电压数据文件

    参数:
        file_path: 电压数据文件的路径
        excluded_nodes: 需要排除的节点列表，默认为None

    返回:
        nodes: 节点名称列表
        voltages: 电压值数组，形状为(n_timesteps, n_nodes)
    """
    # 读取CSV文件
    df = pd.read_csv(file_path)
    
    # 获取时间步列
    steps = df['step'].values
    
    # 获取节点名称列表（不包括'step'列）
    nodes = list(df.columns[1:])
    
    # 如果有需要排除的节点
    if excluded_nodes:
        # 过滤出要保留的节点列
        nodes = [node for node in nodes if node not in excluded_nodes]
    
    # 提取电压数据（不包括'step'列和被排除的节点）
    voltages = df[nodes].values
    
    return nodes, voltages


def visualize_voltage_data(nodes, voltages, filename=None):
    """
    可视化节点电压数据，特别关注接近越限的电压值

    参数:
        nodes: 节点名称列表
        voltages: 电压值数组，形状为(n_timesteps, n_nodes)
        filename: 保存图像的文件名，默认为None（不保存）

    返回:
        Figure: Matplotlib图形对象
    """
    # 时间步数和总节点数
    n_timesteps = voltages.shape[0]
    n_nodes = voltages.shape[1]
    
    # 创建时间轴
    time_steps = np.arange(n_timesteps)
    
    # 设置图表和子图
    fig = plt.figure(figsize=(26 / 2.54, (22 / 2.54) * (8 / 12)))
    
    # 确定有问题的节点
    critical_nodes = []
    normal_nodes = []
    
    for i, node in enumerate(nodes):
        node_voltages = voltages[:, i]
        min_voltage = np.min(node_voltages)
        max_voltage = np.max(node_voltages)
        
        if min_voltage < 0.95 or max_voltage > 1.05:
            critical_nodes.append((node, i, min_voltage, max_voltage))
        else:
            normal_nodes.append((node, i))
    
    # 按最小/最大电压值排序关键节点
    critical_nodes.sort(key=lambda x: (min(x[2], 2-x[3])))
    
    # 分配颜色映射
    cmap = plt.cm.tab20
    color_index = 0
    node_colors = {}
    node_labels = {}
    
    # 首先为关键节点分配颜色
    for node, idx, min_v, max_v in critical_nodes:
        node_colors[node] = cmap(color_index % 20)
        label = node
        if min_v < 0.95:
            label += f" (min={min_v:.3f})"
        if max_v > 1.05:
            label += f" (max={max_v:.3f})"
        node_labels[node] = label
        color_index += 1
    
    # 然后为正常节点分配颜色（但可能不会全部显示在图例中）
    selected_normal_nodes = []
    if len(normal_nodes) > 0:
        # 选择一些典型的正常节点（最多5个）
        step = max(1, len(normal_nodes) // 5)
        selected_indices = list(range(0, len(normal_nodes), step))[:5]
        selected_normal_nodes = [normal_nodes[i] for i in selected_indices]
        
        for node, idx in selected_normal_nodes:
            node_colors[node] = cmap(color_index % 20)
            node_labels[node] = node
            color_index += 1
    
    # 绘制所有正常节点，使用浅灰色
    for node, idx in normal_nodes:
        if (node, idx) in selected_normal_nodes:
            # 选中的正常节点使用分配的颜色
            plt.plot(time_steps, voltages[:, idx], color=node_colors[node], alpha=0.7, linewidth=1.5)
        else:
            # 其他正常节点使用浅灰色
            plt.plot(time_steps, voltages[:, idx], color='lightgray', alpha=0.3, linewidth=1)
    
    # 绘制关键节点，使用彩色并添加标签
    for node, idx, min_v, max_v in critical_nodes:
        plt.plot(time_steps, voltages[:, idx], color=node_colors[node], linewidth=2.5)
    
    # 添加阈值线
    plt.axhline(y=0.95, color='r', linestyle='--', alpha=0.7, linewidth=2, label='低压阈值 (0.95 p.u.)')
    plt.axhline(y=1.05, color='b', linestyle='--', alpha=0.7, linewidth=2, label='高压阈值 (1.05 p.u.)')
    
    # 添加对1.0的参考线
    plt.axhline(y=1.0, color='k', linestyle='-', alpha=0.2)
    
    # 设置坐标轴
    plt.ylabel('节点电压 (p.u.)')
    
    # 设置Y轴范围，确保包括安全下限
    y_min = max(0.90, np.min(voltages) - 0.03)  # 确保至少显示到0.90
    y_max = min(1.15, np.max(voltages) + 0.02)
    plt.ylim(y_min, y_max)
    
    # 设置横轴为实际时刻（每个时间步为15分钟）
    tick_positions = np.arange(0, n_timesteps, 4)
    tick_labels = [f"{j // 4 :02d}:00" if j % 4 == 0 else "" for j in tick_positions]
    plt.xticks(tick_positions, tick_labels, fontsize=8)
    plt.xlabel('时间 (时:分)')
    
    # 创建自定义图例
    legend_elements = []
    
    # 添加阈值线到图例
    legend_elements.append(plt.Line2D([0], [0], color='r', linestyle='--', linewidth=2, label='低压阈值 (0.95 p.u.)'))
    legend_elements.append(plt.Line2D([0], [0], color='b', linestyle='--', linewidth=2, label='高压阈值 (1.05 p.u.)'))
    
    # 添加关键节点到图例
    for node, idx, min_v, max_v in critical_nodes:
        legend_elements.append(plt.Line2D([0], [0], color=node_colors[node], linewidth=2.5, label=node_labels[node]))
    
    # 添加选中的正常节点到图例
    for node, idx in selected_normal_nodes:
        legend_elements.append(plt.Line2D([0], [0], color=node_colors[node], linewidth=1.5, label=node_labels[node]))
    
    # 如果还有其他未显示的正常节点，添加一个代表它们的条目
    if len(normal_nodes) > len(selected_normal_nodes):
        legend_elements.append(plt.Line2D([0], [0], color='lightgray', linewidth=1, label=f'其他正常节点 ({len(normal_nodes) - len(selected_normal_nodes)}个)'))
    
    # # 创建图例，放在图外右侧
    # plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.01, 0.5),
    #            fontsize=8, frameon=True, fancybox=True, shadow=True)
    #
    # # 设置网格
    # plt.grid(True, alpha=0.3)
    #
    # plt.title('节点电压随时间变化')
    #
    # # 调整布局，为右侧的图例留出空间
    # plt.tight_layout()
    # plt.subplots_adjust(right=0.75)

    # 创建图例，放在图内右上角
    plt.legend(handles=legend_elements, loc='upper left',
               fontsize=8, frameon=True, fancybox=True, shadow=True)

    # 设置网格
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图像
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    return fig


def load_pv_data(file_path, rated_power=80):
    """
    加载光伏功率数据并进行最大值归一化处理

    参数:
        file_path: 标幺化光伏功率数据文件路径
        rated_power: 光伏额定功率(kW)，默认80kW

    返回:
        pv_power: 光伏功率数组 (kW)
    """
    # 读取标幺化光伏数据
    try:
        with open(file_path, 'r') as f:
            pv_values = [float(line.strip()) for line in f if line.strip()]
        print(f"成功读取光伏数据，共{len(pv_values)}个数据点")
        
        # 进行最大值归一化处理
        max_value = max(pv_values)
        if max_value > 0:  # 避免除以零
            pv_values = [val / max_value for val in pv_values]
            print(f"已完成光伏数据最大值归一化处理，最大值：{max_value}")
        else:
            print("光伏数据最大值为0，跳过归一化处理")
            
    except Exception as e:
        print(f"读取光伏数据出错: {e}")
        return np.zeros(96)  # 默认返回一天的零值数组（96个15分钟时间段）
    
    # 转换为实际功率 (kW)
    pv_power = np.array(pv_values) * rated_power
    
    return pv_power


def test_results_analysis(results_dir, index=0):
    """
    分析测试结果并生成报告

    Args:
        results_dir (str): 结果目录路径
        index (int): 图片后缀
    """
    if results_dir == '':
        print(f'空字符串，跳过{index}不做分析。')
        return
    # 生成统计数据
    stats = generate_summary_stats(results_dir)

    # 打印统计结果
    print("===== 充电站性能统计 =====")
    print(f"服务车辆数: {stats['served_vehicles']}")
    print(f"完全满足数: {stats['success_vehicles']}")
    print(f"平均满足率: {stats['average_satisfaction']:.2%}")
    print(f"总充电电量: {stats['total_charging_energy']:.2f} kWh")
    print(f"平均单车充电量: {stats['avg_ev_energy']:.2f} kWh")
    print(f"最大总汽车充电功率: {stats['max_charging_power']:.2f} kW")
    print(f"超专变容量运行时长: {stats['overload_hours']:.2f} 小时")
    print(f"总光伏发电量: {stats['total_pv_energy']:.2f} kWh")
    print(f"电压低于0.95p.u.比例: {stats['undervoltage_ratio']:.2f}%")
    print(f"电压高于1.05p.u.比例: {stats['overvoltage_ratio']:.2f}%")

    # 读取充电调度文件
    charging_files = os.path.join(results_dir, 'schedule.csv')
    ev_ids, charging_powers = load_schedule_data(charging_files)

    # 读取储能数据文件
    storage_file = os.path.join(results_dir, 'storage_schedule.csv')
    storage_data = None
    if os.path.exists(storage_file):
        storage_data = load_storage_data(storage_file)
    
    # 读取光伏数据
    pv_file = os.path.join(r'\dssgym\systems',
                           '1-day-900-s-Solar-2-Average-Pad-06.csv')
    pv_data = None
    if os.path.exists(pv_file):
        print(f"找到光伏数据文件: {pv_file}")
        pv_data = load_pv_data(pv_file, rated_power=80)  # 设置额定功率为80kW
    else:
        print(f"警告: 找不到光伏数据文件: {pv_file}")
        # 尝试使用相对路径
        alt_pv_file = os.path.join(os.path.dirname(os.path.dirname(results_dir)), 
                                  'systems', '1-day-900-s-Solar-2-Average-Pad-06.csv')
        if os.path.exists(alt_pv_file):
            print(f"找到替代光伏数据文件: {alt_pv_file}")
            pv_data = load_pv_data(alt_pv_file, rated_power=80)

    # 生成调度矩阵图
    plt.figure()
    fig = visualize_ev_schedule_revised(charging_powers, ev_ids, storage_data, pv_data, transformer_capacity=800)

    # 保存图形
    output_file = os.path.join(os.getcwd(), f'charging_schedule_{index:02}_{datetime.datetime.now().strftime("%Y%m%d")}.png')
    fig.savefig(output_file)
    print(f"充电调度图已保存至: {output_file}")
    
    # 读取并分析电压数据
    voltage_file = os.path.join(results_dir, 'voltages.csv')
    if os.path.exists(voltage_file):
        # 排除一些不需要分析的节点，例如电源节点
        excluded_nodes = ['sourcebus', 'rg60']
        nodes, voltages = load_voltage_data(voltage_file, excluded_nodes)
        
        # 绘制电压图
        voltage_fig = visualize_voltage_data(nodes, voltages)
        
        # 保存电压图
        voltage_output_file = os.path.join(os.getcwd(), f'voltage_profile_{index:02}_{datetime.datetime.now().strftime("%Y%m%d")}.png')
        voltage_fig.savefig(voltage_output_file)
        print(f"节点电压图已保存至: {voltage_output_file}")
    print("==========================")
    return stats


# 示例用法
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent

    # 第一轮测试结果，历史路径，已经更改，如需重新测试需要调整
    # path_list = [
    #     r'D:\LENOVO\Documents\Python\ML\dssgym\results\results_20250501_021615_13Bus_cbat_1000000_01\test_results_20250501_073028',
    #     r'D:\LENOVO\Documents\Python\ML\dssgym\results\results_20250502_005245_13Bus_cbat_s2_1000000_02\test_results_20250502_065405',
    #     r'D:\LENOVO\Documents\Python\ML\dssgym\results\results_20250502_125646_13Bus_1000000_03\test_results_20250502_214808',
    #     r'D:\LENOVO\Documents\Python\ML\dssgym\results\results_20250503_091058_13Bus_cbat_1000000_04\test_results_20250503_123842',
    #     r'D:\LENOVO\Documents\Python\ML\dssgym\results\results_20250503_124337_13Bus_cbat_s2_1000000_05\test_results_20250503_164645',
    #     r'D:\LENOVO\Documents\Python\ML\dssgym\results\results_20250503_165005_13Bus_1000000_06\test_results_20250504_011214',
    #     r'D:\LENOVO\Documents\Python\ML\dssgym\results\results_20250504_012306_13Bus_cbat_1000000_07\test_results_20250504_052230',
    #     r'D:\LENOVO\Documents\Python\ML\dssgym\results\results_20250504_055308_13Bus_cbat_1000000_08\test_results_20250504_093552',
    # ]
    # rules_agent_path = r'D:\LENOVO\Documents\Python\ML\dssgym\results\test_results_rules_20250506_200736_13Bus_cbat'

    # 演示结果路径
    demo_rel_paths = [
        r'results\results_20250511_002004_13Bus_1000000\test_results_20250511_021621',  # 提供最大功率容量，启用BMS -2 演示2
        r'results\results_20250512_215620_13Bus_1000000\test_results_20250513_001904',  # 按电池最大充电功率，不启用BMS -1 演示1
    ]
    demo_path_list = [str((project_root / p).resolve()) for p in demo_rel_paths]
    for i in range(len(demo_path_list)):
        test_results_analysis(demo_path_list[i], i-2)

    # ppo_agent测试结果路径
    path_rel_list = [
        r'results\results_20250509_213723_13Bus_cbat_1000000\test_results_20250510_115601',
        r'results\results_20250510_132849_13Bus_cbat_1000000\test_results_20250510_182929',
        r'results\results_20250510_194947_13Bus_cbat_1000000\test_results_20250511_001130',
        r'results\results_20250513_224309_13Bus_1000000\test_results_20250514_073953',
        r'results\results_20250511_121502_13Bus_cbat_s2_1000000\test_results_20250511_172037',
        r'results\results_20250512_143129_13Bus_cbat_1000000\test_results_20250512_215128',
        r'results\results_20250514_092650_13Bus_1000000\test_results_20250514_183833',
        r'results\results_20250513_002728_13Bus_cbat_s2_1000000\test_results_20250513_041136',
    ]
    path_list = [str((project_root / p).resolve()) for p in path_rel_list]
    for i in range(len(path_list)):
        test_results_analysis(path_list[i], i+1)

    # rules_agent测试结果路径
    rules_agent_path = str((project_root / r'results\test_results_rules_20250513_152400_13Bus_cbat').resolve())
    test_results_analysis(rules_agent_path, 0)

    pass

