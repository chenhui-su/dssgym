# -*- coding: utf-8 -*-
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-DSSGym-Commercial
# Copyright (c) 2025 Su Chenhui
# @File: ppo_agent.py
# @Time: 2025/3/12 下午6:24
# @Author: Gan Mocai
# @Software: PyCharm

"""
主程序
使用SB3的PPO算法训练智能体，使用实现的DSSGym环境进行电力系统仿真。

算法调参：
    1. 自定义PPO网络参数的方式可见 https://kimi.moonshot.cn/chat/cv8ol6kchmtsrgk9q630

Note:
    1. 统一设置各种随机seed为27
    2. 在原始代码执行过程中，保存模型时，未使用cwd则会保存到对应dss文件所在目录；保存图片在dssgym文件夹下的plots目录中
"""

from stable_baselines3 import PPO
from dssgym.env_register import make_env, remove_parallel_dss
from dssgym.reward_monitor_callback import RewardMonitorCallback

import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio  # 兼容旧版接口
import glob

import argparse
import random
import itertools
import os
import multiprocessing as mp
import warnings

import datetime
import time

from dssgym.end_projection import CustomActionWrapper

def parse_arguments():
    parser = argparse.ArgumentParser(description='Argument Parser')
    parser.add_argument('--env_name', default='13Bus')
    parser.add_argument('--seed', type=int, default=27, metavar='N', help='random seed')
    parser.add_argument('--model_seed', type=int, default=27, metavar='N',
                        help='random seed for model initialization')
    parser.add_argument('--num_steps', type=int, default=10000, metavar='N',
                        help='maximum number of steps')
    parser.add_argument('--mode', type=str, default='ppo',
                        help="running mode, ppo, parallel_ppo, episodic_ppo or dss(legacy, deprecated)")
    parser.add_argument('--num_workers', type=int, default=3, metavar='N',
                        help='number of parallel processes')
    parser.add_argument('--use_plot', type=lambda x: str(x).lower() == 'true', default=False)
    parser.add_argument('--plot_path', type=str, default=r'plots', )
    parser.add_argument('--do_testing', type=lambda x: str(x).lower() == 'true', default=False)
    parser.add_argument('--model_path', type=str, default=r'model',  # 纯粹使用该相对路径时，实际会转到对应dss文件所在目录下
                        help="path to save or load the model")
    parser.add_argument('--learning_rate', type=float, default=3e-4,  # 3e-4
                        help="learning rate for PPO")
    parser.add_argument('--n_steps', type=int, default=2048,
                        help="number of steps to run for each environment per update")
    parser.add_argument('--batch_size', type=int, default=64,
                        help="mini-batch size for optimization")
    # 添加测试模式
    parser.add_argument('--test_only', type=lambda x: str(x).lower() == 'true', default=False,
                        help="是否仅测试已训练的模型")
    # 添加命令行参数以便调整是否打印每步信息
    parser.add_argument('--print_step', type=lambda x: str(x).lower() == 'true', default=False,
                        help="测试时是否打印每步信息")
    parser.add_argument('--ev_demand_path', type=str, default=None,
                        help="EV需求csv路径。支持绝对路径或相对于项目根目录的路径。")
    parser.add_argument('--allow_legacy_dss', type=lambda x: str(x).lower() == 'true', default=False,
                        help="是否允许执行已软废弃的 dss 模式（默认 false）")
    arguments = parser.parse_args()
    return arguments


def build_runtime_config(args):
    """构建并向下游环境创建函数透传的运行时配置。"""
    runtime_config = {}
    if args is not None and getattr(args, 'ev_demand_path', None):
        runtime_config['ev_demand'] = args.ev_demand_path
    return runtime_config or None


def seeding(seed) -> None:
    """
    统一设置随机种子，包括numpy、random和os环境变量
    Args:
        seed: 随机种子
    """
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def seeding_all(seed) -> None:
    """
    统一设置随机种子，包括numpy、random、os环境变量和PyTorch
    Args:
        seed: 随机种子
    """
    import torch

    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def test_ppo_agent(model=None, model_path=None, output_dir=None, args=None, load_profile_idx=0,
                   worker_idx=None, use_plot=False, print_step=False, gen_calculate=False):
    """
    测试PPO智能体并保存结果到指定目录

    Args:
        model: 已训练好的PPO模型对象(可选)
        model_path: 模型文件路径(可选)
        output_dir: 输出目录路径(可选)，不指定时使用model_path父目录
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
    if model is None and model_path is None:
        raise ValueError("必须提供model或model_path其中之一！")

    if print_step is None:
        print_step = args.print_step

    # 确定保存路径
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if output_dir:
        # 使用指定的输出目录
        base_dir = output_dir
    elif model_path:
        # 使用模型文件的父目录的父目录
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(model_path)))
    else:
        # 如果都没指定，使用当前工作目录
        base_dir = os.getcwd()

    # 创建结果目录
    save_path = os.path.join(base_dir, f"test_results_{timestamp}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 创建绘图子目录
    plot_dir = os.path.join(save_path, "plots")
    if use_plot and not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # 获取环境
    env = make_env(args.env_name, worker_idx=worker_idx, runtime_config=build_runtime_config(args))
    env.seed(args.seed + 0 if worker_idx is None else worker_idx)
    env = CustomActionWrapper(env)

    # 加载模型(如果未提供model)
    if model is None and model_path is not None:
        from stable_baselines3 import PPO
        try:
            model = PPO.load(model_path, env=env)
            print(f"已从{model_path}加载模型")
        except Exception as e:
            print(f"加载模型失败: {e}")
            return 0, save_path

    # 记录测试信息
    with open(os.path.join(save_path, "test_info.txt"), "w", encoding='utf-8') as f:
        f.write(f"测试时间: {timestamp}\n")
        f.write(f"环境名称: {args.env_name}\n")
        f.write(f"负载配置索引: {load_profile_idx}\n")
        if model_path:
            f.write(f"模型路径: {model_path}\n")
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

    with open(rewards_file, 'w', encoding='utf-8') as f:  # 优化逻辑，便于后续再做修改
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
    obs, info = env.reset(seed=args.seed, options={'load_profile_idx': load_profile_idx})

    episode_reward = 0.0
    for i in range(env.horizon):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_reward += reward

        if print_step:
            print(f"进程 {worker_idx}, Step {i}\n"
                  f"Action: {action}, Obs: {obs}, Reward: {reward:.4f}, Done: {done}, Info: {info}.")

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

        # 记录各节点电压——无法直接通过obs进行，step传出的obs是展为数组的
        with open(voltages_file, 'a') as f:
            line = f"{i}"
            for bus_name in env.all_bus_names:
                # 直接使用环境的circuit对象获取电压数据，类似于环境内部的实现
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
            if isinstance(obs, np.ndarray):
                for obs_val in obs:
                    line += f",{obs_val}"
            else:
                line += f",{obs}"
            f.write(line + "\n")

        # 记录系统功率
        with open(total_powers_file, 'a') as f:
            active_power = env.circuit.total_power()[0]
            reactive_power = env.circuit.total_power()[1]
            power_loss = env.circuit.total_loss()[0]
            power_factor = active_power / (
                np.sqrt(active_power ** 2 + reactive_power ** 2)) if active_power != 0 or reactive_power != 0 else 0
            # 修正功率因数符号
            power_factor = abs(power_factor) if (active_power<0 and reactive_power<=0) or (active_power>0 and reactive_power>=0) else -abs(power_factor)

            f.write(f"{i},{active_power},{reactive_power},{power_loss},{power_factor}\n")

        # 记录EV充电站统计数据
        with open(ev_stats_file, 'a') as f:
            connection_rate = env.obs.get('ev_connection_rate', 0)
            charging_power = env.obs.get('ev_charging_power', 0)
            connected_count = env.obs.get('ev_connected_count', 0)
            success_rate = env.obs.get('ev_success_rate', 0)
            avg_target_achieved = env.obs.get('avg_target_achieved', 0)

            f.write(f"{i},{connection_rate},{charging_power},{connected_count},{success_rate},{avg_target_achieved}\n")

        # 记录各节点的功率 - 从连接总线的负荷和发电设备获取功率
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

                    # 获取连接到该总线的负荷——通过kW和kVar获取得到的是基值而非实际值
                    if env.circuit.dss.ActiveCircuit.Loads.First != 0:
                        while True:
                            # 使用OpenDSS命令获取负荷连接的总线
                            load_name = env.circuit.dss.ActiveCircuit.Loads.Name  # 获取迭代器当前所指的对象的名称
                            env.circuit.dss.Text.Commands(f"? load.{load_name}.bus1")
                            load_bus = env.circuit.dss.Text.Result.split(".")[0]
                            # 检查负载是否连接到当前总线
                            # load_bus = env.circuit.dss.ActiveCircuit.Loads.Bus1.split('.')[0]
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

                    # 获取连接到该总线的电源（其实还包括电池）——在model=1且未设置调度曲线时，kW和实际功率应该一致
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
                                # 使用OpenDSS命令获取负荷连接的总线
                                gen_name = env.circuit.dss.ActiveCircuit.Generators.Name  # 获取迭代器当前所指的对象的名称
                                env.circuit.dss.Text.Commands(f"? generator.{gen_name}.bus1")
                                gen_bus = env.circuit.dss.Text.Result.split(".")[0]
                                # 检查发电机是否连接到当前总线
                                # gen_bus = env.circuit.dss.ActiveCircuit.Generators.Bus1.split('.')[0]
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
            # node_bound参数用于决定为具有多相的节点绘制最大或最小节点电压
            fig, _ = env.plot_graph()
            fig.tight_layout(pad=0.1)
            fig.savefig(os.path.join(plot_dir, f"node_voltage_{i:04d}.png"))
            plt.close()

    # 导出充电站调度表
    try:
        env.ev_station.export_schedule(output_path=os.path.join(save_path, "schedule.csv"))
        env.ev_station.export_storage_statuses(output_path=os.path.join(save_path, "storage_schedule.csv"))
    except Exception as e:
        print(f"导出充电站调度表失败: {e}。")

    # 生成系统布局图和动画
    if use_plot:
        try:
            # 系统布局图
            fig, _ = env.plot_graph(show_voltages=False)
            fig.tight_layout(pad=0.1)
            fig.savefig(os.path.join(plot_dir, "system_layout.pdf"))
            plt.close()

            # 生成动画
            images = []
            filenames = sorted(glob.glob(os.path.join(plot_dir, "node_voltage_*.png")))
            for filename in filenames:
                images.append(imageio.imread(filename))

            if images:  # 确保有图像才生成GIF
                # 使用更好的参数设置来确保动画效果
                imageio.mimsave(
                    os.path.join(plot_dir, 'node_voltage.gif'),
                    images,
                    fps=2,  # 提高帧率使动画更流畅
                    loop=0,  # 0表示无限循环
                    duration=500  # 每帧显示500毫秒
                )
                print(f"已生成GIF动画，包含{len(images)}步")
        except Exception as e:
            print(f"生成图像或动画失败: {e}")

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


def test_saved_model(model_path, args, load_profile_idx=0, use_plot=True, print_step:bool=None):
    """
    测试已保存的模型

    Args:
        model_path: 模型文件路径
        args: 命令行参数
        load_profile_idx: 负载配置索引
        use_plot: 是否绘制图形
        print_step: 是否打印每步详情
    """
    print(f"使用保存的模型 {model_path} 进行测试...")
    if print_step is None:
        print_step = args.print_step
    return test_ppo_agent(
        model_path=model_path,
        args=args,
        load_profile_idx=load_profile_idx,
        use_plot=use_plot,
        print_step=print_step
    )


def run_ppo_agent(args, load_profile_idx=0, worker_idx=None, use_plot=False, print_step:bool=None):
    """
    训练PPO智能体并进行测试
    Args:
        args: 命令行参数
        load_profile_idx: 负载配置索引
        worker_idx: 进程ID
        use_plot: 是否绘图
        print_step: 是否打印每步信息
    """
    cwd = os.getcwd()
    # print('当前工作目录:', cwd)  # 默认运行时工作目录不符合预期

    # 创建以时间戳和环境名命名的文件夹
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"results_{timestamp}_{args.env_name}_{args.num_steps}"
    save_path = os.path.join(cwd, 'results', save_dir)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 在文件夹中创建保存模型和图像子目录
    model_dir = os.path.join(save_path, args.model_path)
    # plot_dir = os.path.join(save_path, args.plot_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    # if use_plot and not os.path.exists(plot_dir):
    #     os.makedirs(plot_dir)

    # 获取环境
    env = make_env(args.env_name, worker_idx=worker_idx, runtime_config=build_runtime_config(args))

    env.seed(args.seed + 0 if worker_idx is None else worker_idx)  # 不同进程使用不同的种子

    env = CustomActionWrapper(env)

    if print_step:
        print('This system has {} capacitors, {} regulators and {} batteries'.format(env.cap_num, env.reg_num,
                                                                                     env.bat_num))
        print('reg, bat action nums: ', env.reg_act_num, env.bat_act_num)
        print('-' * 80)

    # 创建回调
    reward_monitor = RewardMonitorCallback(log_freq=100, output_path=os.path.join(save_path, 'rewards_in_training.csv'))  # 每100步记录一次

    # 创建PPO模型并显式设置随机种子
    model = PPO("MlpPolicy", env,  # 默认策略网络，两层隐藏层的全连接网络，每层包含64个神经元
                learning_rate=args.learning_rate,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                seed=args.model_seed,
                verbose=1)

    # 训练模型
    model.learn(total_timesteps=args.num_steps, callback=reward_monitor)  # 使用回调进行训练

    # 保存模型
    model_path = os.path.join(model_dir, "ppo_model")
    if worker_idx is not None:
        # 为并行工作进程添加唯一标识符
        model_path = f"{model_path}_{worker_idx}"

    model.save(model_path)  # 根据实际情况，好像不必检查目录是否存在，会自动创建
    print(f"模型已保存至 {model_path}")

    # 保存奖励函数权重
    reward_weights_file = os.path.join(save_path, "reward_weights.csv")
    with open(reward_weights_file, 'w', encoding='utf-8') as f:
        f.write("power_w,cap_w,reg_w,soc_w,dis_w,con_w,com_w,energy_w,voltage_w\n")
        f.write(f"{env.reward_func.power_w},{env.reward_func.cap_w},{env.reward_func.reg_w},"
                f"{env.reward_func.soc_w},{env.reward_func.dis_w},{env.reward_func.con_w},{env.reward_func.com_w},{env.reward_func.energy_w},"
                f"{env.reward_func.voltage_w}\n")
    print(f"奖励函数权重已保存至 {reward_weights_file}.")

    # 保存一些训练设置信息
    train_env_settings_info_file = os.path.join(save_path, "train_env_settings_info.txt")
    with open(train_env_settings_info_file, 'w', encoding='utf-8') as f:
        f.write(f"ev_demand_path: {env.ev_demand_path}\n")
        f.write(f"ev_station_bus: {env.ev_station_bus}\n")
        f.write(f"ev_charger_num: {env.ev_charger_num}\n")
        f.write(f"ev_charger_kW: {env.ev_charger_kW}\n")

    # 测试训练得到的模型
    test_ppo_agent(model_path=model_path,args=args,load_profile_idx=load_profile_idx,use_plot=use_plot,print_step=print_step)


def run_parallel_ppo_agent(args):
    """
    并行运行多个PPO智能体
    """
    workers = []
    for i in range(0, args.num_workers):
        p = mp.Process(target=run_ppo_agent, args=(args, i, i))
        p.start()
        workers.append(p)
    for p in workers:
        p.join()

    remove_parallel_dss(args.env_name, args.num_workers)


def ppo_evaluate(model, env, profiles, episodes=None):
    """
    使用PPO模型评估性能

    Args:
        model: PPO模型
        env: 环境
        profiles: 负载配置列表
        episodes: 评估的回合数量

    Returns:
        评估回合的平均回报和标准差
    """
    returns = np.zeros(len(profiles)) if episodes is None else np.zeros(min(episodes, len(profiles)))
    for i in range(len(returns)):
        pidx = profiles[i] if episodes is None else random.choice(profiles)
        obs = env.reset(seed=args.seed, load_profile_idx=pidx)  # 显式传递种子
        episode_reward = 0
        episode_steps = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            next_obs, reward, done, _ = env.step(action)
            episode_reward += reward
            episode_steps += 1
            mask = 1 if episode_steps == env.horizon else float(not done)
            obs = next_obs
        returns[i] = episode_reward
    return returns.mean(), returns.std()


def run_episodic_ppo_agent(args, worker_idx=None):
    """运行基于回合的PPO智能体，包含训练-测试分割

    Args:
        args: 命令行参数
        worker_idx: 工作进程ID
    """
    test_profiles = None
    fout = None
    # 输出文件
    if args.do_testing:
        fout = open("result/{}_ppo_{}.csv".format(args.env_name, args.seed), 'w')

    # 获取环境
    env = make_env(args.env_name, worker_idx=worker_idx, runtime_config=build_runtime_config(args))
    env.seed(args.seed + 0 if worker_idx is None else worker_idx)

    # 获取观察和动作空间
    obs_dim = env.observation_space.shape[0]
    CRB_num = (env.cap_num, env.reg_num, env.bat_num)
    CRB_dim = (2, env.reg_act_num, env.bat_act_num)
    print('NumCap, NumReg, NumBat: {}'.format(CRB_num))
    print('ObsDim, ActDim: {}, {}'.format(obs_dim, sum(CRB_num)))
    print('-' * 80)

    # 训练-测试分割
    if args.do_testing:
        train_profiles = random.sample(range(env.num_profiles), k=env.num_profiles // 2)
        test_profiles = [i for i in range(env.num_profiles) if i not in train_profiles]
    else:
        train_profiles = list(range(env.num_profiles))

    # 创建PPO模型并显式设置随机种子
    model = PPO("MlpPolicy", env,
                learning_rate=args.learning_rate,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                seed=args.model_seed,
                verbose=1)

    # 训练循环
    total_num_steps = 0

    for i_episode in itertools.count(start=1):
        # 每个回合训练模型一定步数
        model.learn(total_timesteps=env.horizon)
        total_num_steps += env.horizon

        # 评估当前回合的性能
        load_profile_idx = random.choice(train_profiles)
        obs = env.reset(seed=args.seed, load_profile_idx=load_profile_idx)  # 显式传递种子
        episode_reward = 0
        episode_steps = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            next_obs, reward, done, info = env.step(action)
            episode_steps += 1
            episode_reward += reward
            obs = next_obs

        print(f"episode: {i_episode}, profile: {load_profile_idx},"
              f" total num steps: {total_num_steps}, episode steps: {episode_steps},"
              f" reward: {episode_reward:.2f}")

        if total_num_steps >= args.num_steps:
            # 保存最终模型
            model.save(args.model_path)
            print(f"模型已保存至 {args.model_path}")
            break

        # 定期在测试集上评估
        if args.do_testing and i_episode % 5 == 0:
            avg_reward, std = ppo_evaluate(model, env, test_profiles)
            fout.write('{},{},{}\n'.format(total_num_steps, avg_reward, std))
            fout.flush()
            print("----------------------------------------")
            print("Avg., Std. Reward: {}, {}".format(round(avg_reward, 2), round(std, 2)))
            print("----------------------------------------")


def run_dss_agent(args):
    """运行 legacy dss 基线路径（已软废弃，仅保留兼容）。"""
    deprecate_msg = (
        "`--mode dss` is legacy and soft-deprecated in DSSGym because EV-battery behavior "
        "is not fully represented by OpenDSS internal controllers."
    )
    warnings.warn(deprecate_msg, DeprecationWarning, stacklevel=2)
    print(f"[DEPRECATED] {deprecate_msg}")
    if not getattr(args, 'allow_legacy_dss', False):
        print("[DEPRECATED] 已跳过 dss 模式执行。若需继续，请显式传入 `--allow_legacy_dss true`。")
        return

    # 获取环境
    env = make_env(args.env_name, dss_act=True, runtime_config=build_runtime_config(args))
    env.seed(args.seed)

    profiles = list(range(env.num_profiles))

    # 训练循环
    total_num_steps = 0  # 追踪所有回合的累计步数

    for i_episode in itertools.count(start=1):
        episode_reward = 0
        episode_steps = 0
        done = False
        load_profile_idx = random.choice(profiles)
        reset_result = env.reset(seed=args.seed, options={'load_profile_idx': load_profile_idx})
        if isinstance(reset_result, tuple):
            obs = reset_result[0]
        else:
            obs = reset_result

        while not done:
            step_result = env.dss_step()
            if isinstance(step_result, tuple) and len(step_result) == 5:
                next_obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            elif isinstance(step_result, tuple) and len(step_result) == 4:
                next_obs, reward, done, info = step_result
            else:
                raise RuntimeError(f"Unexpected dss_step return format: {type(step_result)}")
            episode_steps += 1
            total_num_steps += 1
            episode_reward += reward
            obs = next_obs

        print(f"episode: {i_episode}, profile: {load_profile_idx}, total num steps: {total_num_steps}, "
              f"episode steps: {episode_steps}, reward: {episode_reward:.2f}")
        if total_num_steps >= args.num_steps:
            break


if __name__ == '__main__':
    args = parse_arguments()
    seeding(args.seed)
    if args.test_only:
        if not os.path.exists(args.model_path):
            print(f"错误: 模型文件 {args.model_path} 不存在")
        else:
            test_saved_model(args.model_path, args, load_profile_idx=0, use_plot=args.use_plot)
    elif args.mode.lower() == 'ppo':
        start_time = time.time()
        run_ppo_agent(args, worker_idx=None, use_plot=args.use_plot, print_step=False)
        stop_time = time.time()
        execution_time = stop_time - start_time
        print(f"运行时间: {execution_time:.2f}秒")
    elif args.mode.lower() == 'parallel_ppo':
        run_parallel_ppo_agent(args)
    elif args.mode.lower() == 'episodic_ppo':
        run_episodic_ppo_agent(args)
    elif args.mode.lower() == 'dss':
        run_dss_agent(args)
    else:
        raise NotImplementedError("运行模式未实现")
