# -*- coding: utf-8 -*-
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-DSSGym-Commercial
# Copyright (c) 2025 Su Chenhui
# @File: reward_curve.py
# @Time: 2025/5/4 19:53
# @Author: Gan Mocai
# @Software: PyCharm, VS Code

"""
根据历史奖励数据的csv绘制奖励曲线，包括：
    1. 训练奖励曲线——回合奖励和历史平均奖励，数据来源csv文件，路径示例：D:\LENOVO\Documents\Python\ML\powergym\results_20250503_165005_13Bus_1000000\rewards_in_training.csv
    2. 测试奖励曲线——各奖励函数子项曲线和总曲线，数据来源csv文件，路径示例：D:\LENOVO\Documents\Python\ML\powergym\results_20250503_165005_13Bus_1000000\test_results_20250504_011214\rewards.csv
"""

import argparse
import datetime

import pandas as pd
import matplotlib.pyplot as plt

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

def plot_training_reward(csv_path, index:int=0):
    """

    Args:
        csv_path: 训练过程奖励数据的CSV文件路径。
        index: 图片后缀，便于区分，防止覆盖。

    Returns:
        None
    """
    df = pd.read_csv(csv_path)

    # 确保数据类型正确（将step和reward列转换为数值类型）
    df["step"] = pd.to_numeric(df["step"], errors="coerce") # 其实是episode
    df["reward"] = pd.to_numeric(df["reward"], errors="coerce")

    # 使用窗口大小100计算移动平均回合奖励
    window = 100
    # 在DataFrame上直接计算移动平均
    df['rewards_ma'] = df['reward'].rolling(window=window, min_periods=1).mean()

    episodes = df["step"].tolist()  # 需要时再转换为列表
    rewards = df["reward"].tolist()
    rewards_ma = df["rewards_ma"].tolist()

    # print(episodes, rewards, rewards_ma)

    plt.figure(figsize=(16/2.54, 10/2.54))
    plt.plot(episodes, rewards, 'b-', alpha=0.3, label="单回合奖励")
    plt.plot(episodes, rewards_ma, 'r-', linewidth=2, label="移动平均奖励")
    # 自动调整Y轴范围，使图像更清晰
    # 这里使用极低分位和最大值作为显示范围的经验裁剪，并非 5% / 95% 分位
    reward_q1 = pd.Series(rewards).quantile(0.000005)
    reward_q3 = pd.Series(rewards).quantile(1)
    y_min = max(reward_q1 / 1.5, min(rewards_ma) / 1.2)  # 确保能显示移动平均线
    y_max = min(reward_q3 * 1.5, max(rewards_ma) * 1.2)  # 确保能显示移动平均线
    plt.ylim(y_min, y_max)

    plt.xlabel("回合")
    plt.ylabel("奖励")
    plt.title("训练过程奖励曲线")
    plt.legend(loc='upper left', borderaxespad=0)
    plt.grid(True)
    plt.savefig(f'training_reward_curve_{index:02d}_{datetime.datetime.now().strftime("%Y%m%d")}.png', dpi=300)  # 保存图像
    plt.show()


def plot_test_reward(csv_path, index:int=0):
    """
    根据测试结果CSV文件绘制奖励曲线。
    Args:
        csv_path: 测试结果CSV文件的路径，包含各奖励函数分项及总奖励数据。
        index: 图片后缀，便于区分，防止覆盖。

    Returns:
        None
    """
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(10, 6))
    # 假设CSV包含各奖励函数分项及总奖励数据，绘制所有曲线
    for col in df.columns:
        plt.plot(df.index, df[col], label=col)
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.title("Test Reward Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(f'test_reward_curve_{index:02d}_{datetime.datetime.now().strftime("%Y%m%d")}.png', dpi=300)  # 保存图像
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="根据CSV数据绘制奖励曲线")
    parser.add_argument("mode", choices=["train", "test"], help="选择绘制训练或测试奖励曲线")
    parser.add_argument("csv_paths", nargs='+',
                        help="奖励数据CSV文件的路径，如：path/to/rewards_in_training.csv，可输入多个路径")
    parser.add_argument("--indices", type=int, nargs='*',
                        help="图表索引值，与CSV路径一一对应，不指定时自动从0开始编号")
    args = parser.parse_args()

    # 处理索引值，如果未提供或提供的数量不足，则自动补充
    indices = args.indices if args.indices else []
    if len(indices) < len(args.csv_paths):
        # 如果有索引，则从最后一个索引值开始递增；否则从0开始
        start_idx = indices[-1] + 1 if indices else 0
        indices.extend(range(start_idx, start_idx + len(args.csv_paths) - len(indices)))
        print(f"警告：索引数量不足，已自动从{start_idx}开始补充索引至: {indices}")

    for i, csv_path in enumerate(args.csv_paths):
        if args.mode == "train":
                plot_training_reward(csv_path, indices[i])
        elif args.mode == "test":
                plot_test_reward(csv_path, indices[i])


if __name__ == "__main__":
    main()
    # # 1st 轮 8 个算例
    # path_list = [
    #     r'D:\LENOVO\Documents\Python\ML\dssgym\results_20250501_021615_13Bus_cbat_1000000_01\rewards_in_training.csv',
    #     r'D:\LENOVO\Documents\Python\ML\dssgym\results_20250502_005245_13Bus_cbat_s2_1000000_02\rewards_in_training.csv',
    #     r'D:\LENOVO\Documents\Python\ML\dssgym\results_20250502_125646_13Bus_1000000_03\rewards_in_training.csv',
    #     r'D:\LENOVO\Documents\Python\ML\dssgym\results_20250503_091058_13Bus_cbat_1000000_04\rewards_in_training.csv',
    #     r'D:\LENOVO\Documents\Python\ML\dssgym\results_20250503_124337_13Bus_cbat_s2_1000000_05\rewards_in_training.csv',
    #     r'D:\LENOVO\Documents\Python\ML\dssgym\results_20250503_165005_13Bus_1000000_06\rewards_in_training.csv',
    #     r'D:\LENOVO\Documents\Python\ML\dssgym\results_20250504_012306_13Bus_cbat_1000000_07\rewards_in_training.csv',
    #     r'D:\LENOVO\Documents\Python\ML\dssgym\results_20250504_055308_13Bus_cbat_1000000_08\rewards_in_training.csv',
    # ]
    # # 2nd 轮 8 个算例
    # path_list = [
    #     r'D:\LENOVO\Documents\Python\ML\powergym\results_20250509_213723_13Bus_cbat_1000000\rewards_in_training.csv',
    #     r'D:\LENOVO\Documents\Python\ML\powergym\results_20250510_132849_13Bus_cbat_1000000\rewards_in_training.csv',
    #     r'D:\LENOVO\Documents\Python\ML\powergym\results_20250510_194947_13Bus_cbat_1000000\rewards_in_training.csv',
    #     r'D:\LENOVO\Documents\Python\ML\powergym\results_20250513_224309_13Bus_1000000\rewards_in_training.csv',
    #     r'D:\LENOVO\Documents\Python\ML\powergym\results_20250511_121502_13Bus_cbat_s2_1000000\rewards_in_training.csv',
    #     r'D:\LENOVO\Documents\Python\ML\powergym\results_20250512_143129_13Bus_cbat_1000000\rewards_in_training.csv',
    #     r'D:\LENOVO\Documents\Python\ML\powergym\results_20250514_092650_13Bus_1000000\rewards_in_training.csv',
    #     r'D:\LENOVO\Documents\Python\ML\powergym\results_20250513_002728_13Bus_cbat_s2_1000000\rewards_in_training.csv',
    # ]
    # for index, path in enumerate(path_list):
    #     plot_training_reward(path, index+1)

