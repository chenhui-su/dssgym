# -*- coding: utf-8 -*-
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-DSSGym-Commercial
# Copyright (c) 2025 Su Chenhui
# @File: ev_demand.py
# @Time: 2025/4/8 15:54
# @Author: Gan Mocai
# @Software: PyCharm

"""
生成EV充电需求队列和模型
"""

import numpy as np
import pandas as pd
import os
import argparse
from collections import deque
import matplotlib.pyplot as plt

# 定义中英文字体映射，鸠占鹊巢
plt.rcParams['font.family'] = ['serif', 'sans-serif']
# 设置英文和数字字体为 Times New Roman
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
# 设置中文字体为宋体
plt.rcParams['font.sans-serif'] = ['SimSun'] + plt.rcParams['font.sans-serif']
# 解决坐标轴负数的负号显示问题
plt.rcParams['axes.unicode_minus'] = False

# 其他字体属性设置
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 16

# # Vehicle daily coupling probability distribution from scidata
# probability_couple_scidata = np.array([
#     2.090694935217903, 2.0318021201413425, 2.090694935217903, 2.0318021201413425,
#     1.9729093050647823, 1.9729093050647823, 1.8551236749116609, 1.707891637220259,
#     1.707891637220259, 1.5606595995288575, 1.4134275618374559, 1.3839811542991753,
#     1.3839811542991753, 1.4134275618374559, 1.3839811542991753, 1.325088339222615,
#     1.2073027090694937, 1.1484098939929328, 1.1484098939929328, 1.1484098939929328,
#     1.1484098939929328, 1.1778563015312131, 1.2367491166077738, 1.5312131919905771,
#     1.884570082449941, 2.1201413427561837, 2.2673733804475855, 2.3262661955241457,
#     2.6207302709069493, 3.2096584216725557, 3.5630153121319204, 3.53356890459364,
#     3.3274440518256774, 3.0624263839811543, 3.651354534746761, 4.446407538280329,
#     4.181389870435806, 3.9752650176678443, 4.446407538280329, 4.5053003533568905,
#     4.976442873969376, 5.5948174322732624, 5.535924617196701, 5.565371024734982,
#     5.889281507656066, 6.095406360424029, 6.27208480565371, 7.1554770318021195,
#     7.567726737338044, 8.00942285041225, 7.803297997644288, 7.243816254416961,
#     6.654888103651355, 6.3309776207302715, 6.654888103651355, 7.773851590106007,
#     8.00942285041225, 7.420494699646643, 7.714958775029446, 7.656065959952886,
#     7.06713780918728, 7.449941107184924, 7.773851590106007, 7.567726737338044,
#     7.6855123674911665, 7.06713780918728, 7.4793875147232045, 7.4793875147232045,
#     8.097762073027091, 7.832744405182568, 7.803297997644288, 7.744405182567727,
#     7.332155477031802, 7.449941107184924, 7.0082449941107186, 6.9493521790341575,
#     6.625441696113074, 6.389870435806831, 5.830388692579505, 6.24263839811543,
#     6.036513545347468, 5.8598351001177855, 5.889281507656066, 5.30035335689046,
#     5.506478209658422, 5.565371024734982, 5.270906949352179, 4.534746760895171,
#     4.446407538280329, 4.269729093050648, 3.769140164899882, 3.4746760895170787,
#     3.032979976442874, 2.502944640753828, 2.237926972909305, 1.884570082449941])
#
# 海口某公共停车场EV耦合的单日时间概率分布
probability_couple_Haikou = np.array([
    11.017236594616262, 11.530634269680244, 12.182788073139896, 12.61447380592643,
    11.398045080324382, 9.620424902099844, 8.658382411889859, 8.209737596743857,
    7.83817951959545, 7.354074804970553, 6.908513459344455, 6.772840800468688,
    6.601708242114027, 6.35194721100182, 6.19006506120687, 5.997348216212883,
    5.741420246060867, 5.582621565785822, 5.567204218186303, 5.721377694181493,
    6.156146896487928, 6.9393481545434925, 7.65317134840122, 8.345410255619623,
    9.160987943634177, 9.6666769448984, 9.999691653048009, 10.522339736671702,
    10.685763621226604, 10.696555764546268, 10.215534519441276, 8.929727729641392,
    6.729672227190034, 4.819462859609633, 3.886713329838735, 3.538281274089606,
    3.6924547500847957, 4.033178132034164, 4.475656008140359, 4.72695877401252,
    4.861089698128334, 4.9320094970861215, 4.848755820048719, 4.728500508772471,
    4.657580709814684, 4.611328667016126, 4.52499152045882, 4.585119176096944,
    4.763960408251364, 4.870340106688046, 5.184853997718233, 5.426906355030681,
    5.610372791464957, 5.770713206499954, 5.7506706546205795, 5.668958712343128,
    5.567204218186303, 5.659708303783417, 5.844716474977645, 6.094477506089852,
    6.532330177916192, 6.837593660386668, 6.637168141592921, 5.57337115722611,
    3.8188770004008514, 2.6024482747988036, 2.2833091794887608, 2.099842743054485,
    1.9795874317782367, 1.9348771237396318, 1.8547069162221332, 1.7822453825043938,
    1.7190342573463662, 1.6558231321883383, 1.584903333230551, 1.6203632327094448,
    1.649656193148531, 1.7020751749868952, 1.7899540563041536, 1.8192470167432397,
    1.7745367087046344, 1.705158644506799, 1.6604483364681941, 1.6157380284295892,
    1.5633190465912246, 1.5263174123523788, 1.4430637353149764, 1.4137707748758903,
    1.418395979155746, 1.4245629181955537, 1.3952699577564676, 1.3073910764392094,
    1.2565138293607967, 1.2595972988807005, 1.2410964817612777, 0.9697511640097437
])

# prob_truncated = probability[:-16]  # 截断概率分布以简化问题
# prob_truncated /= np.sum(prob_truncated)  # 归一化


def generate_power_profile(duration, power_rated=120, curve_type=0):
    """简化版充电曲线生成函数"""
    power_profile = np.zeros(duration)
    if curve_type == 0:
        pre_phase = 1
        trickle_phase = 1
        main_phase = duration - pre_phase - trickle_phase

        power_profile[0] = power_rated * 0.75
        power_profile[1:1 + main_phase] = power_rated
        power_profile[-1] = power_rated * 0.25

    elif curve_type == 1:
        power_profile[0] = power_rated * 0.9
        decay_start = int(duration * 0.7)
        power_profile[1:decay_start] = power_rated
        if duration > decay_start:
            power_profile[decay_start:] = np.linspace(power_rated, power_rated * 0.5, duration - decay_start)

    else:
        power_profile[:] = power_rated

    return power_profile


def generate_charging_requirements(num_ev, power_support, type_curve, battery_capacity=None):
    """
    根据支持的功率和电池容量生成每辆车所需的充电时间

    参数:
    num_ev: 车辆数量
    power_support: 每辆车支持的充电功率（字典）
    battery_capacity: 电池容量字典，如果为None则根据功率生成

    返回:
    required_charging_time: 每辆车所需的充电时间（以时间段计）
    battery_capacity: 电池容量字典
    """
    required_charging_time = {}

    # 根据功率生成电池容量，使得恒功率充电时间在30分钟到1小时之间
    if battery_capacity is None:
        battery_capacity = {}
        for i in range(num_ev):
            # 计算使得充电时间在0.5-1小时之间的电池容量
            # 容量 = 功率 × 时间(小时)
            charging_duration_hours = np.random.uniform(0.5, 1.0)  # 0.5-1小时之间
            battery_capacity[i] = power_support[i] * charging_duration_hours

    # 计算所需充电时间（每个时间段为15分钟）
    for i in range(num_ev):
        # 随机生成需要充入的电量比例（60%-80%）
        charge_percentage = np.random.uniform(0.6, 0.8)
        energy_needed = battery_capacity[i] * charge_percentage

        # 根据充电曲线类型应用不同系数
        power_coefficient = 1.0
        if type_curve[i] == 0:  # 假设类型0是恒功率
            power_coefficient = 1.0
        elif type_curve[i] == 1:  # 假设类型1是递减功率
            power_coefficient = 0.85  # 平均功率降低
        elif type_curve[i] == 2:  # 假设类型2是先增后减
            power_coefficient = 0.9  # 平均功率略有降低

        # 计算充满所需的小时数 = 容量(kWh) / (功率(kW) * 系数)
        hours_needed = energy_needed / (power_support[i] * power_coefficient)

        # 转换为时间段数（离散为15分钟的倍数）
        time_slots_raw = hours_needed * 4  # 4个15分钟时间段 = 1小时

        # 离散化为15分钟的倍数 (1=15min, 2=30min, 3=45min, ...)
        # 向上取整到最近的整数
        time_slots_needed = max(1, int(np.ceil(time_slots_raw)))

        required_charging_time[i] = time_slots_needed

    return required_charging_time, battery_capacity


def load_distribution_from_csv(file_path):
    """
    从CSV文件加载概率分布

    参数:
    file_path: CSV文件路径

    返回:
    values: 可能的取值列表
    probabilities: 对应的概率分布数组
    """
    try:
        df = pd.read_csv(file_path)

        # 检查CSV文件格式并处理
        if 'Probability' in df.columns:
            # 提取非零概率的行
            df_valid = df[df['Probability'] > 0]

            # 获取第一列名称(可能是SOC或时间等)
            value_column = df.columns[0]

            # 获取对应值和概率
            values = df_valid[value_column].values
            probabilities = df_valid['Probability'].values

            # 归一化确保概率和为1
            probabilities = probabilities / np.sum(probabilities)

            return values, probabilities
        else:
            print(f"错误: {file_path} 中未找到'Probability'列")
            return None, None
    except Exception as e:
        print(f"加载分布文件时出错: {e}")
        return None, None


def generate_ev_demand(num_evs, base_path, seed=None):
    """
    根据概率分布生成EV充电需求

    参数:
    num_evs: 生成的电动汽车数量
    base_path: 分布文件所在的基础路径
    seed: 随机种子，用于重现结果

    返回:
    包含EV充电需求信息的字典
    """
    if seed is not None:
        np.random.seed(seed)

    # 加载分布数据
    arrival_values, arrival_dist = load_distribution_from_csv(os.path.join(base_path, 'arrival_distribution.csv'))
    duration_values, duration_dist = load_distribution_from_csv(os.path.join(base_path, 'duration_distribution.csv'))
    start_soc_values, start_soc_dist = load_distribution_from_csv(os.path.join(base_path, 'start_soc_distribution.csv'))
    end_soc_values, end_soc_dist = load_distribution_from_csv(os.path.join(base_path, 'end_soc_distribution.csv'))

    # 如果任一分布文件加载失败，使用默认分布
    if arrival_dist is None:
        print("使用默认耦合概率分布")
        arrival_values = np.arange(96)
        arrival_dist = probability_couple_Haikou

    if duration_dist is None:
        print("使用默认停留时间分布")
        # 默认停留时间分布 - 三种时长的概率
        duration_values = np.array([1, 2, 3])
        duration_dist = np.array([0.7, 0.2, 0.1])

    if start_soc_dist is None:
        print("使用默认起始SOC分布")
        # 默认起始SOC - 10%到40%，步长1%
        start_soc_values = np.linspace(0.1, 0.4, 31)[:-1]  # 0.1到0.39
        start_soc_dist = np.ones(30) / 30

    if end_soc_dist is None:
        print("使用默认目标SOC分布")
        # 默认目标SOC - 80%到100%，步长1%
        end_soc_values = np.linspace(0.8, 1.0, 21)[:-1]  # 0.8到0.99
        end_soc_dist = np.ones(20) / 20

    # 生成EV数据
    ev_data = {}

    # 使用功率水平分布
    power_levels = [60, 80, 100, 120]
    power_probs = [0.3, 0.3, 0.2, 0.2]  # 各功率水平的概率

    for i in range(num_evs):  # 按单辆生成
        # 生成到达时间
        arrive_time = min(int(np.random.choice(arrival_values, p=arrival_dist)), 96-1)  # 避免最后一个时间段的来车

        # 生成停留时间
        stay_duration = np.random.choice(duration_values, p=duration_dist)
        depart_time = min(arrive_time + stay_duration, 96)  # 最大不超过96

        # 生成电池容量和充电功率
        power_support = np.random.choice(power_levels, p=power_probs)

        # 随机电池容量 (40-115 kWh)，与车型（缺少数据）和充电功率相关
        # 功率越大，电池容量往往越大
        min_capacity = 40 + ((power_support - 60) / 20) * 10  # 功率每增加20kW，最小容量增加10kWh
        max_capacity = 70 + ((power_support - 60) / 20) * 15  # 功率每增加20kW，最大容量增加15kWh
        battery_capacity = int(np.random.uniform(min_capacity, max_capacity))
        # battery_capacity = int(round(battery_capacity / 10) * 10)  # 改为整十

        # 生成起始和目标SOC
        start_soc = np.random.choice(start_soc_values, p=start_soc_dist) / 100  # 转换为0-1范围
        end_soc = np.random.choice(end_soc_values, p=end_soc_dist) / 100  # 转换为0-1范围

        # 确保终止SOC大于起始SOC，同时小于1
        end_soc = min(max(end_soc, start_soc + 0.2), 1)

        # 截取SOC
        start_soc = round(start_soc, 2)
        end_soc = round(end_soc, 2)

        # 计算充电能量需求
        energy_demand = battery_capacity * (end_soc - start_soc)

        # 充电曲线类型
        if power_support >= 100:
            curve_type = np.random.choice([0, 1, 2], p=[0.8, 0.1, 0.1])
        elif power_support == 80:
            curve_type = np.random.choice([0, 1, 2], p=[0.1, 0.8, 0.1])
        else:
            curve_type = np.random.choice([0, 1, 2], p=[0.1, 0.1, 0.8])

        # 计算所需的充电时间（以15分钟时段计）
        power_coefficient = 1.0 if curve_type == 0 else (0.85 if curve_type == 1 else 0.9)
        hours_needed = energy_demand / (power_support * power_coefficient)
        time_slots_needed = max(1, int(np.ceil(hours_needed * 4)))  # 每小时4个时间段

        # 存储EV数据
        ev_data[i] = {
            'arrive_time': arrive_time,
            'depart_time': depart_time,
            'power_support': power_support,
            'battery_capacity': battery_capacity,
            'start_soc': start_soc,
            'end_soc': end_soc,
            'energy_demand': energy_demand,
            'curve_type': curve_type,
            'time_slots_needed': time_slots_needed
        }

    return ev_data


def generate_ev_parameters(Num_TimeSteps, Num_EVs, prob):
    """
    旧的生成汽车参数数据

    参数:
      Num_TimeSteps: 时间段总数
      Num_EVs: 车辆数量
      prob: 到达时间概率分布（原使用截断的耦合概率，截断最后四小时，重新归一化）

    返回:
      arrival, departure, power_support, curve_type, required_charging_time, battery_capacity
    """
    np.random.seed(22)  # in conference paper 22
    arrival = np.random.choice(range(Num_TimeSteps - 16), size=Num_EVs, p=prob)  # 按照概率分布生成车辆到达时间
    stay_hours = np.random.choice([1, 2, 3], Num_EVs, p=[0.7, 0.2, 0.1])  # 随机的停留时间分布
    departure = {i: min(arrival[i] + stay_hours[i] * 4, Num_TimeSteps) for i in range(Num_EVs)}
    # 使用离散值生成充电功率 (60, 80, 100, 120 kW)
    power_levels = [60, 80, 100, 120]
    power_support = {i: np.random.choice(power_levels) for i in range(Num_EVs)}
    curve_type = {i: np.random.randint(0, 3) for i in range(Num_EVs)}  # 随机分配充电曲线类型
    # 将充电能量时间需求转换为充电时间需求
    required_charging_time, battery_capacity = generate_charging_requirements(Num_EVs, power_support, curve_type)
    return arrival, departure, power_support, curve_type, required_charging_time, battery_capacity


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="生成 EV 充电需求队列")
    parser.add_argument(
        "--base_path",
        type=str,
        default=os.getenv("DSSGYM_EV_DISTRIBUTION_PATH"),
        help="分布文件目录，支持绝对路径或相对当前目录路径。也可通过环境变量 DSSGYM_EV_DISTRIBUTION_PATH 设置。",
    )
    args = parser.parse_args()

    if not args.base_path:
        raise ValueError("缺少 --base_path。请传入分布文件目录，或设置 DSSGYM_EV_DISTRIBUTION_PATH。")

    # 基本数量参数
    Num_EVs = 250  # 车辆数 in conference paper 250
    Num_TimeSteps = 96  # 时间段数
    Power_Station = 500  # 充电站功率限制
    Num_Chargers = 10  # 最大充电桩数

    # 汽车参数生成
    # 1. 分布文件基础路径
    base_path = args.base_path
    dist_scenario = os.path.basename(base_path)
    dist_location, dist_day_type = os.path.basename(base_path).split('-')

    # 2. 生成EV需求数据
    np.random.seed(22)  # 设置随机种子 in conference paper 22
    ev_demand = generate_ev_demand(Num_EVs, base_path)

    # 从生成的需求数据中提取必要信息用于后续处理
    arrive_time = {i: ev_demand[i]['arrive_time'] for i in range(Num_EVs)}
    depart_time = {i: ev_demand[i]['depart_time'] for i in range(Num_EVs)}
    power_support = {i: ev_demand[i]['power_support'] for i in range(Num_EVs)}
    curve_type = {i: ev_demand[i]['curve_type'] for i in range(Num_EVs)}
    charge_duration_required = {i: ev_demand[i]['time_slots_needed'] for i in range(Num_EVs)}
    battery_capacity = {i: ev_demand[i]['battery_capacity'] for i in range(Num_EVs)}

    # 打印统计信息
    print(f"根据 {dist_scenario} 场景的分布生成了 {Num_EVs} 辆电动汽车的充电需求")
    print(f"平均电池容量: {np.mean(list(battery_capacity.values())):.2f} kWh")
    print(f"平均充电时长: {np.mean(list(charge_duration_required.values())) / 4:.2f} 小时")

    # 3. 转换为DataFrame并保存为CSV文件
    ev_df = pd.DataFrame.from_dict(ev_demand, orient='index')
    csv_file = os.path.join(os.getcwd(), f"ev_demand-{dist_scenario}-{Num_EVs}-A95.csv")
    ev_df.to_csv(csv_file, index=False)
    print(f"EV需求数据已保存至 {csv_file}")
