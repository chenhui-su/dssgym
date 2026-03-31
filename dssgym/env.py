# Copyright 2025 Su Chenhui
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-DSSGym-Commercial

"""
基于 OpenDSS 的电力系统强化学习环境（Env）。它提供了一个完整的接口，
用于管理电力系统的状态、执行控制动作、计算奖励，并支持系统可视化。
环境原支持电容器、调压器和电池的联合控制，以实现电压调节、功率损耗优化和储能管理等多目标优化任务。

主要类和函数：
- ActionSpace: 定义电力系统控制器的动作空间（离散/连续）
- Env: 核心环境类，实现强化学习环境的标准接口
    - MyReward: 定义复合奖励函数，包括电压违规、功率损耗和控制成本还有充电站相关内容
- 辅助函数：系统状态可视化、电池位置选择等

设计用于与现代强化学习算法（如PPO）集成，支持单步和多步控制策略优化。

Improve:
    1. 修改风格，翻译注释，修改动作空间为仅控制电池功率——保留环境空间中的电容、调压器状态（以便未来恢复），仅动作空间不做动作。
    2. 添加ev充电需求接口，充电站管理模块引用，相应统计添加在其中。

Note: 修改对应位置
    状态空间在step()和reset_obs_space()中修改，并需要在wrap_obs()中补充适配（可能还有其他地方）；
        observe_load的启用和关闭在初始化和reset_obs_space()中进行；
    奖励函数在MyReward类中修改，需要依靠对应的obs或info信息；
    动作空间在ActionSpace类中修改，并需要在step和Env.init修改适配；
        动作空间的修改还需要在battery的功率映射中中进行调整。
    拓扑设置在对应daily的dss文件中修改；
    时间步修改在dss和env_register中修改；

曾使用monkey patch临时替换函数reset，参考自 https://yuanbao.tencent.com/chat/naQivTmsDa/26ca9f85-b738-4e39-8d3d-7550578fdb25
"""
import logging

import gymnasium as gym
from gymnasium.utils import seeding

from .circuit import Circuits, BatteryController, load_ev_from_csv, BatteryStationManager
from .loadprofile import LoadProfile
import networkx as nx

import os
import numpy as np
import matplotlib.pyplot as plt

import math  # 单纯为向上取整引入，有点亏


# %% 辅助函数
def plotting(env, profile, episode_step, show_voltages=True) -> None:
    """ 使用负载配置在某一回合步骤中绘制网络状态

    Args:
        env (obj): 环境对象
        profile (int): 负载配置编号
        episode_step (int): 回合中的步骤编号
        show_voltages (bool): 是否显示电压
    """
    cwd = os.getcwd()
    if not os.path.exists(os.path.join(cwd, 'plots')):
        os.makedirs(os.path.join(cwd, 'plots'))

    fig, _ = env.plot_graph(show_voltages=show_voltages)
    fig.tight_layout(pad=0.1)
    fig.savefig(os.path.join(cwd, 'plots/' + str(profile).zfill(3) + '_' + str(episode_step) + '.png'))
    plt.close()


def farthest_first_traversal_selection(vio_nodes, dist_matrix, k=10) -> list[str]:
    """
    从电压违规的节点中使用最远优先遍历算法来选择电池。Farthest first traversal to select batteries from the violated nodes.

    Arguments:
        vio_nodes (list): 电压标幺值低于0.95的母线名称
        dist_matrix (np.array): 违规节点的成对距离矩阵 the pairwise distance matrix of the violated nodes
        k (int): 电池数

    Returns:
        被选中节点的名称列表 list of the names of the chosen nodes

    Raises:
        Error: 当k<2时抛出。
    """
    assert k > 1, 'invalid k(k<2)'
    if len(vio_nodes) <= 1:
        return vio_nodes

    # 对于>=2个违规节点
    # 随机初始化点
    chosen = [np.random.randint(len(vio_nodes))]

    # 构建 dist_map 和第二个点
    dist_map = dict()
    max_dist = p = 0
    for i in range(len(vio_nodes)):
        if i != chosen[-1]:
            dist = dist_matrix[i, chosen[-1]]
            dist_map[i] = dist
            if dist > max_dist:
                max_dist = dist
                p = i
    del dist_map[p]
    chosen.append(p)

    for kk in range(2, k):
        if len(dist_map) == 0:
            break

        # 更新 'dist_map' 和 'p'
        max_dist = p = 0
        for pt, val in dist_map.items():
            dist = min(val, dist_matrix[pt, chosen[-1]])
            if dist < val:
                dist_map[pt] = dist
            if dist > max_dist:
                max_dist = dist
                p = pt
        del dist_map[p]
        chosen.append(p)
    return [vio_nodes[c] for c in chosen]


def choose_batteries(env, k=10, on_plot=True, node_bound='minimum') -> list[str]:
    """
    选择电池的 locations

    Arguments:
        env (obj): 环境对象
        k (int): 要分配的电池数量
        on_plot (bool): 在 pos 中显示的节点上分配电池
        node_bound (str): Determine to plot max/min node voltage for nodes with more than one phase

    Returns:
        nodes被选择节点(nodes)的名称列表(list)
    """
    assert node_bound in ['minimum', 'maximum'], 'invalid node_bound'

    graph = nx.Graph()
    graph.add_edges_from(list(env.lines.values()) + list(env.transformers.values()))
    lens = dict(nx.shortest_path_length(graph))

    if node_bound == 'minimum':
        nv = {bus: min(volts) for bus, volts in env.obs['bus_voltages'].items()}
    else:
        nv = {bus: max(volts) for bus, volts in env.obs['bus_voltages'].items()}

    if on_plot:
        _, pos = env.plot_graph(show_voltages=False)
        nv = {bus: volts for bus, volts in nv.items() if bus in pos}

    vio_nodes = [bus for bus, vol in nv.items() if vol < 0.95]
    dist = np.zeros((len(vio_nodes), len(vio_nodes)))
    for i, b1 in enumerate(vio_nodes):
        for j, b2 in enumerate(vio_nodes):
            dist[i, j] = lens[b1][b2]

    choice = farthest_first_traversal_selection(vio_nodes, dist, k)
    return choice


def get_basekv(env, buses) -> list[str]:
    """获取指定母线的基准电压，打印并返回

    Args:
        env: 环境对象
        buses: 指定电压的名称数组

    Returns:

    """
    # buses = ['l3160098', 'l3312692', 'l3091052', 'l3065696', 'l3235247', 'l3066804', 'l3251854', 'l2785537', 'l2839331', 'm1069509']
    ans = []
    for busname in buses:
        env.circuit.dss.Circuits.SetActiveBus(busname)
        ans.append(env.circuit.dss.Circuits.Buses.kVBase)
    print(f'指定母线基准电压为{ans}')
    return ans


# %% 动作空间
class ActionSpaceV0:
    """电容器、调压器和电池的动作空间封装

    Attributes:
        cap_num (int): 电容器的数量
        reg_num (int): 调压器的数量
        bat_num (int): 电池的数量
        reg_act_num (int): 调压器控制动作数量
        bat_act_num (int): 电池控制动作数量
        space (gym.spaces): 来自gym的空间对象  对于电池的动作部分：如果使用离散电池，space是MultiDiscrete类型；否则，space是MultiDiscrete和Box的元组
    """

    def __init__(self, CRB_num, RB_act_num):
        self.cap_num, self.reg_num, self.bat_num = CRB_num
        self.reg_act_num, self.bat_act_num = RB_act_num

        if self.bat_act_num < float('inf'):
            # 离散
            self.space = gym.spaces.MultiDiscrete([2] * self.cap_num + [self.reg_act_num] * self.reg_num +
                                                  [self.bat_act_num] * self.bat_num)
        else:
            # 连续
            self.space = gym.spaces.Tuple((gym.spaces.MultiDiscrete([2] * self.cap_num +
                                                                    [self.reg_act_num] * self.reg_num),
                                           gym.spaces.Box(low=-1, high=1, shape=(self.bat_num,))))

    def sample(self) -> np.ndarray:
        """返回从动作空间采样的一个随机动作.

        Returns:
        np.ndarray: 随机动作向量

        """
        ss = self.space.sample()  # tuple[Any, ...]
        if self.bat_act_num == np.inf:
            return np.concatenate(ss)
        return ss

    def seed(self, seed) -> None:
        self.space.seed(seed)

    def dim(self) -> int:
        if self.bat_act_num == np.inf:
            return self.space[0].shape[0] + self.space[1].shape[0]
        return self.space.shape[0]

    def CRB_num(self):
        return self.cap_num, self.reg_num, self.bat_num

    def RB_act_num(self):
        return self.reg_act_num, self.bat_act_num


class ActionSpace:
    """电池充电功率的动作空间封装。移除电容和变压器抽头的动作。

    Attributes:
        bat_num (int): 电池的总数量
        sto_num (int): 储能电池的数量
        bat_act_num (int): 每个电池控制动作的数量
        space (gym.spaces): 来自gymnasium的空间对象 (如果使用离散电池，space是MultiDiscrete类型；否则，space是Box类型)
    """

    def __init__(self, bat_num, sto_num, bat_act_num):
        self.bat_num = bat_num
        self.bat_act_num = bat_act_num

        if self.bat_act_num < float('inf'):
            # 离散
            # self.space = gym.spaces.MultiDiscrete([self.bat_act_num] * self.bat_num)
            self.space = gym.spaces.MultiDiscrete(
                [self.bat_act_num] * sto_num + [(self.bat_act_num // 2) + 1] * (self.bat_num - sto_num))
        else:
            # 连续
            # self.space = gym.spaces.Box(low=-1, high=1, shape=(self.bat_num,),
            #                             dtype=np.float32)  # To avoid or confine V2G, 改变 high，因为是发电机，然后我懒得改.
            # 修改以避免V2G
            high = np.zeros(self.bat_num)
            high[:sto_num] = 1.0
            self.space = gym.spaces.Box(low=-1.0, high=high, dtype=np.float32)


    def sample(self) -> np.ndarray:
        """返回从动作空间采样的一个随机动作.

        Returns:
            np.ndarray: 随机动作向量
        """
        return self.space.sample()

    def seed(self, seed) -> None:
        self.space.seed(seed)

    def dim(self) -> int:
        return self.space.shape[0]

    def CRB_num(self):
        return 0, 0, self.bat_num

    def RB_act_num(self):
        return 0, self.bat_act_num


# %% environment class
class Env(gym.Env):
    """用于训练强化学习智能体的环境

    Attributes:
        obs (dict): 系统的观察/状态
        dss_folder_path (str): 包含DSS文件的文件夹路径
        dss_file (str): DSS仿真文件名
        source_bus (str): 最接近电源的母线（在BusCoords.csv中有坐标）
        node_size (int): 绘图中节点的大小
        shift (int): 绘图中标签的偏移量
        show_node_labels (bool): 是否在绘图中显示节点标签
        scale (float): 负载曲线的缩放比例
        wrap_observation (bool): 是否在reset和step输出中将obs展平为numpy数组
        observe_load (bool): 是否在观察中包含节点负载

        load_profile (obj): 负载曲线管理类
        num_profiles (int): 由load_profile生成的不同曲线数量
        horizon (int): 每个回合的最大步数
        circuit (obj): 连接到DSS仿真的电路对象
        all_bus_names (list): 系统中所有母线名称
        cap_names (list): 电容器母线列表
        reg_names (list): 调压器母线列表
        bat_names (list): 电池母线列表
        cap_num (int): 电容器数量
        reg_num (int): 调压器数量
        bat_num (int): 电池数量
        reg_act_num (int): 调压器控制动作数量
        bat_act_num (int): 电池控制动作数量
        topology (graph): 电力系统的NxGraph
        reward_func (obj): 奖励函数类
        t (int): 环境状态的时间步
        ActionSpace (obj): 动作空间类，用于随机抽样动作
        action_space (gym.spaces): 来自ActionSpace类的基础动作空间
        observation_space (gym.spaces): 环境的观察空间

    在self.step()和self.reset()中定义:
        all_load_profiles (dict): 所有母线和时间的负载曲线2D数组

    在self.step()中定义并在self.plot_graph()中使用:
        str_action: 在self.plot_graph()中打印的动作字符串

    在self.build_graph()中定义:
        edges (dict): 连接电路中节点的边的字典
        lines (dict): 电路中带组件的边的字典
        transformers (dict): 系统中变压器的字典
    """

    def __init__(self, folder_path, info, dss_act=False):
        """
        利用dss文件和环境信息初始化环境
        Args:
            folder_path: 包含DSS文件的文件夹路径
            info: 包含系统信息和环境信息的字典
            dss_act: 是否使用 OpenDSS 内部控制机制，而不是代理的控制动作。
                当 dss_act=True 时：
                    环境使用 OpenDSS 内置控制器来管理电容器和调压器
                    环境会在 dss_step() 方法中使用这些内置控制器（783-831行）
                    环境观察 OpenDSS 控制器做出的改变并据此计算奖励
                当 dss_act=False 时（默认值）：
                    代理完全控制所有组件
                    环境使用标准的 step() 方法（在代码中被 step_battery() 替换）
                    代理的动作直接应用于电路组件
                这个参数提供了一种方式来比较代理控制策略与 OpenDSS 内置控制方法的效果，或在某些测试场景中使用 OpenDSS 的内部控制。
        """
        super().__init__()
        self.obs = dict()
        self.dss_folder_path = os.path.join(folder_path, info['system_name'])
        self.dss_file = info['dss_file']
        self.source_bus = info['source_bus']
        self.node_size = info['node_size']
        self.shift = info['shift']
        self.show_node_labels = info['show_node_labels']
        self.scale = info['scale'] if 'scale' in info else 1.0
        self.wrap_observation = True
        self.observe_load = True  # 设置观察负荷，启用后从loadprofile截取负载曲线包括在obs中
        self.lines = None  # 连接电路中节点的边的字典

        # 生成负荷 profile files
        self.load_profile = LoadProfile(
            info['max_episode_steps'],
            self.dss_folder_path,
            self.dss_file,
            worker_idx=info['worker_idx'] if 'worker_idx' in info else None)

        self.num_profiles = self.load_profile.gen_loadprofile(scale=self.scale)  # 生成负荷曲线
        # 选择一个dummy(虚拟) load profile用户电路初始化
        self.load_profile.choose_loadprofile(0)

        # 问题的horizon是the length of load profile
        self.horizon = info['max_episode_steps']  # horizon指在交互过程中的可用步数，即最大步数。
        self.reg_act_num = info['reg_act_num']
        self.bat_act_num = info['bat_act_num']  # 从info字典中获取
        assert self.horizon >= 1, 'invalid horizon'
        assert self.reg_act_num >= 2 and self.bat_act_num >= 2, 'invalid act nums'

        # 创建电路对象（初始化过程已经进行编译）
        self.circuit = Circuits(os.path.join(str(self.dss_folder_path), str(self.dss_file)),
                                RB_act_num=(self.reg_act_num, self.bat_act_num),
                                dss_act=dss_act)
        self.all_bus_names = self.circuit.dss.ActiveCircuit.AllBusNames
        # 从电路中提取信息
        self.cap_names = list(self.circuit.capacitors.keys())
        self.reg_names = list(self.circuit.regulators.keys())
        self.bat_names = list(self.circuit.batteries.keys())
        self.cap_num = len(self.cap_names)
        self.reg_num = len(self.reg_names)
        self.bat_num = len(self.bat_names)  # 获取储能电池数量
        self.sto_num = self.bat_num # 记录储能电池的数量

        # Note: 添加内容
        self.con_num = info['num_chargers']  # 获取充电桩数量
        self.bat_num = self.con_num + len(self.bat_names)  # 从参数中获取EV连接点数量，再加上电路中已设置的数量
        self.station_bus = info['bus_name']  # 充电站连接的母线名称
        print(f"环境初始化中，充电站母线为 {self.station_bus}.")
        # 创建充电站
        self.station_transformer_capacity = info['transformer_kVA']  # 专变容量
        self.ev_demand_path = info['ev_demand']  # 用于保存到训练信息
        self.ev_station_bus = info['bus_name']   # 用于保存到训练信息
        self.ev_charger_num = info['num_chargers']  # 用于保存到训练信息
        self.ev_charger_kW = info['charger_kW'] # 用与保存到训练信息
        self.ev = load_ev_from_csv(info['ev_demand'])  # EV信息
        self.ev_controller = BatteryController(self.circuit)
        self.circuit.battery_controller = self.ev_controller
        self.ev_station = BatteryStationManager(self.circuit, self.station_bus, self.con_num, self.ev_charger_kW,
                                                self.ev['arrival'],self.ev['departure'],
                                                self.ev['max_power'] ,self.ev['initial_soc'], self.ev['target_soc'],
                                                self.ev['capacity'],self.ev['curve_type'])
        self.circuit.ev_station = self.ev_station  # 将充电站添加到电路引用

        # 检查参数范围
        assert self.cap_num >= 0 and self.reg_num >= 0 and self.bat_num >= 0 and \
               self.cap_num + self.reg_num + self.bat_num >= 1, 'invalid CRB_num'

        self.topology = self.build_graph()
        self.reward_func = self.MyReward(self, info)
        self.t = 0

        # 创建 action space 和 observation space
        # self.ActionSpace = ActionSpace((self.cap_num, self.reg_num, self.bat_num),
        #                                (self.reg_act_num, self.bat_act_num))
        self.ActionSpace = ActionSpace(self.bat_num, self.sto_num, self.bat_act_num)  # 修改为仅控制电池
        self.action_space = self.ActionSpace.space  # 标准化接口
        self.str_action = None  # 动作字符串，用于在self.plot_graph()打印
        self.reset_obs_space()

    def reset_obs_space(self, load_profile_idx=0, wrap_observation=True, observe_load=True):
        """reset the observation space 基于 option of wrapping and load. 此处设置观察空间的维度数和上下限。

        instead of setting directly from the attribute (e.g., Env.wrap_observation)
        it is suggested to set wrap_observation and observe_load through this function
        Args:
            load_profile_idx (int): 负载配置索引
            wrap_observation (bool): 是否将观察展平为数组
            observe_load (bool): 是否在观察中包含节点负载
        """
        self.wrap_observation = wrap_observation
        self.observe_load = observe_load

        # self.reset(load_profile_idx=0)
        obs, info = self.reset(seed=27, options={'load_profile_idx': load_profile_idx})
        # nnode = len(self.obs['bus_voltages'])  # 统计电压数
        nnode = len(np.hstack(list(self.obs['bus_voltages'].values())))
        if observe_load:
            nload = len(self.obs['load_profile_t'])

        if self.wrap_observation:
            low, high = [0.8] * nnode, [1.2] * nnode  # add voltage bound  节点电压标幺值
            logging.debug(f"\n电压边界维度: {nnode}")
            low, high = low + [0] * self.cap_num, high + [1] * self.cap_num  # add cap bound 电容投切
            logging.debug(f"电容边界维度: {self.cap_num}")
            low, high = low + [0] * self.reg_num, high + [self.reg_act_num] * self.reg_num  # add reg bound 抽头
            logging.debug(f"Regulator边界维度: {self.reg_num}")
            low, high = low + [0, -1] * self.bat_num, high + [1, 1] * self.bat_num  # add bat bound (SOC, 功率)
            logging.debug(f"电池边界维度: {self.bat_num * 2}")

            # 添加充电站实时指标维度
            dim_before = len(low)
            low, high = low + [0.0, 0.0, 0.0, 0.0, 0.0], high + [1.0, float('inf'), float('inf'), 1.0, float('inf')]  # 连接率,充电功率,总连接数,成功率,平均能量满足率
            logging.debug(f"EV station metrics dimension: {len(low)-dim_before}")

            if observe_load:
                low, high = low + [0.0] * nload, high + [1.0] * nload  # add load bound
                logging.debug(f"Load 边界维度: {nload}")
            low, high = np.array(low, dtype=np.float32), np.array(high, dtype=np.float32)
            self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)  # Note: 设置为np.float32试试水
            # 总维度数
            logging.info(f"观察空间维度设置为: {len(low)}")
        else:
            bat_dict = {bat: gym.spaces.Box(np.array([0, -1]), np.array([1, 1]), dtype=np.float32)
                        for bat in self.obs['bat_statuses'].keys()}
            obs_dict = {
                'bus_voltages': gym.spaces.Box(0.8, 1.2, shape=(nnode,)),
                'cap_statuses': gym.spaces.MultiDiscrete([2] * self.cap_num),
                'reg_statuses': gym.spaces.MultiDiscrete([self.reg_act_num] * self.reg_num),  # 修正为self.reg_num，原本为cap_num
                'bat_statuses': gym.spaces.Dict(bat_dict),
                # 添加充电站指标
                'ev_connection_rate': gym.spaces.Box(0.0, 1.0, shape=(1,)),
                'ev_charging_power': gym.spaces.Box(0.0, float('inf'), shape=(1,)),
                'ev_connected_count': gym.spaces.Box(0.0, float('inf'), shape=(1,)),
                'ev_success_rate': gym.spaces.Box(0.0, 1.0, shape=(1,)),
                'avg_target_achieved': gym.spaces.Box(0.0, float('inf'), shape=(1,))
            }
            if observe_load:
                obs_dict['load_profile_t'] = gym.spaces.Box(0.0, 1.0, shape=(nload,))
            self.observation_space = gym.spaces.Dict(obs_dict)

    class MyReward:
        """奖励函数定义类（位置在环境类中以简化代码，类名不影响功能），定义了多种奖励函数，包括功率损耗、电压违规和控制成本等。
        
        Attributes:
            env (obj): 继承环境的所有属性
        """

        def __init__(self, env, info):
            self.env = env
            self.power_w = info['power_w']
            self.cap_w = info['cap_w']
            self.reg_w = info['reg_w']
            self.soc_w = info['soc_w']
            self.dis_w = info['dis_w']
            self.com_w = info['completion_w']
            self.con_w = info['connection_w']
            self.energy_w = info['energy_w']
            self.voltage_w = info['voltage_w']
            self.tf_capacity_w = info['tf_capacity_w']
            # 添加组成部分列表，便于后续记录
            self.components = ['PowerLoss_reward', 'Voltage_reward', 'Control_reward',
                             'Connection_reward', 'Completion_reward', 'Energy_reward', 'Transformer_reward']

        def powerloss_reward(self):
            # Penalty for power loss of entire system at one time step
            loss = self.env.circuit.total_loss()[0]  # a postivie float
            gen = self.env.circuit.total_power()[0]  # a negative float
            ratio = max(0.0, min(1.0, self.env.obs['power_loss']))  # 功率损耗和总功率的比值
            return -ratio * self.power_w

        def ctrl_reward(self, capdiff, regdiff, soc_err, discharge_err):
            # 动作惩罚，调节cap和reg会有惩罚，最终SOC小于最初的SOC也会给出一个惩罚。
            # capdiff: abs(current_cap_state - new_cap_state)
            # regdiff: abs(current_reg_tap_num - new_reg_tap_num)
            # soc_err: abs(soc - initial_soc)
            # discharge_err: max(0, kw) / max_kw
            # discharge_err > 0 means discharging
            cost = self.cap_w * sum(capdiff) + \
                   self.reg_w * sum(regdiff) + \
                   (0.0 if self.env.t != self.env.horizon else self.soc_w * sum(soc_err)) + \
                   self.dis_w * sum(discharge_err)
            return -cost

        def completion_reward(self):
            # 完成率奖励
            return max(0.0, self.env.obs['ev_success_rate']) * self.com_w

        def connection_reward(self):
            # 连接数奖励
            return self.env.obs['ev_connected_count'] * self.con_w

        def energy_reward(self):
            # 能量满足率奖励
            return self.env.obs['avg_target_achieved'] * self.energy_w

        def voltage_reward(self, record_node=False):
            # Penalty for node voltage being out of [0.95, 1.05] range
            violated_nodes = []
            total_violation = 0
            for name, voltages in self.env.obs['bus_voltages'].items():
                max_penalty = min(0, 1.05 - max(voltages))  # penalty is negative if above max
                min_penalty = min(0, min(voltages) - 0.95)  # penalty is negative if below min
                total_violation += (max_penalty + min_penalty)
                if record_node and (max_penalty != 0 or min_penalty != 0):
                    violated_nodes.append(name)
            return total_violation*self.voltage_w, violated_nodes

        def tf_reward(self):
            # 变压器裕度奖励，超出容量上限则给出巨额惩罚
            tf_kVA = self.env.station_transformer_capacity
            ev_kW = self.env.obs['ev_charging_power']
            ev_kVar = ev_kW * np.tan(np.arccos(self.env.ev_station.EV_PF))
            storage_kW = sum([bat.actual_power() for name, bat in self.env.circuit.storage_batteries.items() if
                              name in self.env.bat_names[:self.env.sto_num]])
            storage_kVar = sum(
                [bat.actual_power() * np.tan(np.arccos(bat.pf)) for name, bat in self.env.circuit.storage_batteries.items() if
                 name in self.env.bat_names[:self.env.sto_num]])
            PV_kW = sum(load.feature[1] for key, load in self.env.circuit.loads.items() if "PV" in key and load.bus == self.env.ev_station_bus)  # feature0 kV,1 kW, 2 kVar
            PV_kVar = sum(load.feature[2] for key, load in self.env.circuit.loads.items() if "PV" in key and load.bus == self.env.ev_station_bus)
            total_kW = ev_kW + storage_kW + PV_kW
            total_kVar = ev_kVar + storage_kVar + PV_kVar
            total_kVA = np.sqrt(total_kW ** 2 + total_kVar ** 2)
            remain_kVA = 0.75 * tf_kVA - total_kVA
            if total_kVA > tf_kVA:
                logging.warning("超出专变容量上限")
                return 1000 * self.tf_capacity_w * remain_kVA
            elif total_kVA > 0.75 * tf_kVA:
                logging.warning("超出专变安全限值")
                return self.tf_capacity_w * remain_kVA
            else:
                return 0

        def composite_reward(self, cd, rd, soc, dis, full=True, record_node=False):
            # the main reward function
            powerloss = self.powerloss_reward()
            voltage, vio_nodes = self.voltage_reward(record_node)
            ctrl = self.ctrl_reward(cd, rd, soc, dis)
            connection = self.connection_reward()
            completion = self.completion_reward()  # 添加充电完成率奖励
            energy = self.energy_reward()
            transformer = self.tf_reward()

            total_reward = powerloss + voltage + ctrl + connection + completion + energy + transformer

            info = dict() if not record_node else {'violated_nodes': vio_nodes}
            if full:
                info.update({'power_loss_ratio': -powerloss / self.power_w,
                             'PowerLoss_reward': powerloss, 'Voltage_reward': voltage, 'Control_reward': ctrl,
                             'Connection_reward': connection, 'Completion_reward': completion, 'Energy_reward': energy, 'Transformer_reward': transformer})

            logging.info(f"奖励各项：{powerloss=}, {voltage=}, {ctrl=}, {connection=}, {completion=}, {energy=}, {transformer=}.")

            return total_reward, info

    def step(self, action: np.ndarray):
        """
        已添加排队逻辑在其中，控制逻辑在相应调用内修改。
        Args:
            action (array): 电池的动作数组
        Returns:
            self.wrap_obs(self.obs): 环境的下一个观察值(array)
            reward: 执行动作获得的奖励(float)
            terminated: 回合是否因完成条件而终止(bool)
            truncated: 回合是否因时间限制等原因被截断(bool)
            info: 额外信息(dict)
        """
        action_idx = 0
        self.str_action = ''  # the action string to be printed at self.plot_graph()

        # 匹配空间维度，改起来太麻烦
        if self.reg_num > 0:
            # 保留当前调压器状态
            # 由于不改变调压器状态，差值为0
            reg_statuses = {name: reg.tap for name, reg in self.circuit.regulators.items()}
            regdiff = [0] * self.reg_num
        else:
            regdiff, reg_statuses = [], dict()
        if self.cap_num > 0:
            # 保留当前电容器状态
            # 由于不改变电容器状态，差值为0
            cap_statuses = {name: cap.status for name, cap in self.circuit.capacitors.items()}
            capdiff = [0] * self.cap_num
        else:
            capdiff, cap_statuses = [], dict()

        # 队列处理
        # 原本顺序（1次检查到达，1次等待队列处理）没有问题，不会出现新到达车辆比排队车辆更先接入，首次调用时不存在排队队列，后续如果队列非空，不会空桩.但是检查到达在检查离开之后，可能就会存在这种情况。
        self.ev_station.process_waiting_queue()  # 在上一轮尾这一轮初的离开检查后，处理等待队列，以免出现后到先接入的情况。
        self.ev_station.check_arrivals()  # 检查到达
        self.ev_station.process_waiting_queue()  # 处理等待队列

        print(f"智能体的动作为: {action}")

        # 控制电池功率
        if self.bat_num > 0:
            states = action  # 整个 action 向量都是电池功率控制信号
            self.circuit.set_all_batteries_before_solve(states)  # 具体控制
            self.str_action += 'Bat Status:' + str(states)

        # 运行电路仿真
        self.circuit.dss.ActiveCircuit.Solution.Solve()

        # 更新电池信息 kWh. record soc_err and discharge_err
        if self.bat_num > 0:
            soc_errs, dis_errs = self.circuit.set_all_batteries_after_solve()
            # bat_statuses = {name: [bat.soc, -1 * bat.actual_power() / bat.max_kw] for name, bat in self.circuit.batteries.items()}
            # 创建包含所有电池的状态字典
            bat_statuses = {}
            # 首先添加电路中定义的储能电池
            for name, bat in self.circuit.storage_batteries.items():
                bat_statuses[name] = [bat.soc, -1 * bat.actual_power() / bat.max_kw]

            # 然后添加充电站中连接的EV电池
            if hasattr(self.circuit, 'ev_station') and self.ev_station is not None:
                ev_statuses = self.ev_station.get_all_statuses()
                for idx, status in enumerate(ev_statuses):
                    # 为充电站的每个连接点创建一个虚拟电池状态
                    bat_statuses[f'charger_{idx:02d}'] = status
        else:
            soc_errs, dis_errs, bat_statuses = [], [], dict()

        logging.debug(f'当前在执行step函数，电池状态键值对的数量为{len(bat_statuses)}.')  # 通过输出检查电池接入情况
        # 更新统计数据，以免被离开检查提前释放，没记录上。
        # 更新总功率
        self.ev_station.update_statistics(update_type="summary")

        # 更新功率调度方案记录
        self.ev_station.update_schedule()
        self.ev_station.update_storage_statuses()

        # 更新时间步
        self.t += 1
        # 对齐时间步
        self.ev_station.current_step = self.t

        # 队列处理 对于每个时段，离开先于到达。
        self.ev_station.check_departures()  # 检查离开

        # 更新 obs
        bus_voltages = dict()
        for bus_name in self.all_bus_names:
            bus_voltages[bus_name] = self.circuit.bus_voltage(bus_name)
            bus_voltages[bus_name] = [bus_voltages[bus_name][i] for i in range(len(bus_voltages[bus_name])) if
                                      i % 2 == 0]

        self.obs['bus_voltages'] = bus_voltages
        self.obs['cap_statuses'] = cap_statuses
        self.obs['reg_statuses'] = reg_statuses
        self.obs['bat_statuses'] = bat_statuses
        self.obs['power_loss'] = - self.circuit.total_loss()[0] / self.circuit.total_power()[0]  # 功率损耗和总功率的比值
        self.obs['time'] = self.t

        # 加入充电站的状态信息——实时信息
        if hasattr(self, 'ev_station') and self.ev_station is not None:
            self.obs['ev_connection_rate'] = self.ev_station.stats['current_connection_rate']
            self.obs['ev_charging_power'] = self.ev_station.stats['current_charging_power']
            self.obs['ev_connected_count'] = self.ev_station.stats['connected_count']
            self.obs['ev_success_rate'] = self.ev_station.stats['current_success_rate']
            self.obs['avg_target_achieved'] = self.ev_station.stats['avg_target_achieved']
        else:
            self.obs['ev_connection_rate'] = 0.0
            self.obs['ev_charging_power'] = 0.0
            self.obs['ev_connected_count'] = 0
            self.obs['ev_success_rate'] = 0.0
            self.obs['avg_target_achieved'] = 0.0

        if self.observe_load:
            self.obs['load_profile_t'] = self.all_load_profiles.iloc[self.t % self.horizon].to_dict()  # 截取负载曲线包括在obs中

        # 检查异常或偏大的obs值
        self._check_extreme_values(self.obs)

        # 为兼容gymnasium的更新要求: split 'done' into 'terminated' and 'truncated'
        terminated = (self.t == self.horizon)  # Episode is done due to termination condition
        truncated = False  # Not truncated due to time limit, etc.

        # 计算奖励函数并返回额外信息
        reward, info = self.reward_func.composite_reward(capdiff, regdiff, soc_errs[:self.sto_num], dis_errs)

        # noinspection PyTypeChecker
        info.update({'av_cap_err': sum(capdiff) / (self.cap_num + 1e-10),
                     'av_reg_err': sum(regdiff) / (self.reg_num + 1e-10),
                     'av_dis_err': sum(dis_errs) / (self.bat_num + 1e-10),
                     'av_soc_err': sum(soc_errs) / (self.bat_num + 1e-10),
                     'av_soc': sum([soc for _, [soc, _] in bat_statuses.items()]) / (
                             self.bat_num + 1e-10)})  # avoid dividing by zero

        if self.wrap_observation:
            return self.wrap_obs(self.obs), reward, terminated, truncated, info
        else:
            return self.obs, reward, terminated, truncated, info

    def _check_extreme_values(self, obj, key=""):
        """递归检查异常值"""
        if isinstance(obj, dict):
            for k, v in obj.items():
                self._check_extreme_values(v, f"{key}.{k}" if key else k)
        elif isinstance(obj, (list, np.ndarray)):
            try:
                arr = np.array(obj, dtype=float)
                if np.any(~np.isfinite(arr)) or np.any(np.abs(arr) > 1e10):
                    logging.warning(f"检测到极端值 '{key}': {obj}")
            except (TypeError, ValueError):
                # 无法转换为数值数组，跳过
                pass
        elif isinstance(obj, (int, float)):
            if not np.isfinite(obj) or abs(obj) > 1e10:
                logging.warning(f"检测到极端值 '{key}': {obj}")

    def reset(self, *, seed: int | None = None, options: dict[str, any] | None = None):
        """重置环境状态以开始新的回合，符合gymnasium的更新要求，添加了重置充电站

        进行的重置操作包括：重置时间步、选择负载配置文件、重新编译DSS、重置电池状态、更新母线电压、电容器状态、调压器状态、电池状态和功率损耗等。

        Args:
            seed: 随机数生成器的种子
            options: 重置的额外选项
                load_profile_idx: 负载配置文件的ID号（默认: 0）

        Returns:
            tuple: 包含以下元素的元组
                observation: 封装后的观察值
                info: 额外信息
        """
        # Initialize the RNG if the seed is manually passed 初始化随机数生成器，如果手动传递种子
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)

        # Extract load_profile_idx from options or use default
        load_profile_idx = 0
        if options and 'load_profile_idx' in options:
            load_profile_idx = options['load_profile_idx']

        # reset time
        self.t = 0

        # 重置充电站——确保obs的bat_statuses能正常工作
        if hasattr(self, 'ev_station') and self.ev_station is not None:
            self.ev_station.reset()

        # choose load profile
        self.load_profile.choose_loadprofile(load_profile_idx)
        self.all_load_profiles = self.load_profile.get_loadprofile(load_profile_idx)

        # re-compile dss and reset batteries
        self.circuit.reset()

        # node voltages
        bus_voltages = dict()
        for bus_name in self.all_bus_names:
            bus_voltages[bus_name] = self.circuit.bus_voltage(bus_name)
            bus_voltages[bus_name] = [bus_voltages[bus_name][i] for i in range(len(bus_voltages[bus_name])) if
                                      i % 2 == 0]
        self.obs['bus_voltages'] = bus_voltages

        # status of capacitor
        cap_statuses = {name: cap.status for name, cap in self.circuit.capacitors.items()}
        self.obs['cap_statuses'] = cap_statuses

        # status of regulator
        reg_statuses = {name: reg.tap for name, reg in self.circuit.regulators.items()}
        self.obs['reg_statuses'] = reg_statuses

        # status of battery
        # bat_statuses = {name: [bat.soc, -1 * bat.actual_power() / bat.max_kw] for name, bat in
        #                 self.circuit.batteries.items()}  # 原本
        bat_statuses = {}
        # 添加电路中定义的静态储能电池
        for name, bat in self.circuit.storage_batteries.items():
            bat_statuses[name] = [bat.soc, -1 * bat.actual_power() / bat.max_kw]

        # 添加充电站EV的初始状态
        if hasattr(self, 'ev_station') and self.ev_station is not None:
            ev_statuses = self.ev_station.get_all_statuses()
            for idx, status in enumerate(ev_statuses):
                bat_statuses[f'charger_{idx:02d}'] = status
        self.obs['bat_statuses'] = bat_statuses

        # total power loss
        self.obs['power_loss'] = -self.circuit.total_loss()[0] / self.circuit.total_power()[0]

        # time step tracker
        self.obs['time'] = self.t

        # load for current timestep
        if self.observe_load:
            self.obs['load_profile_t'] = self.all_load_profiles.iloc[self.t].to_dict()

        # Edge weight
        # self.obs['Y_matrix'] = self.circuit.edge_weight

        # Prepare info dictionary
        info = {
            'load_profile_idx': load_profile_idx
        }

        # 重置充电站的状态信息
        if hasattr(self, 'ev_station') and self.ev_station is not None:
            self.obs['ev_connection_rate'] = self.ev_station.stats['current_connection_rate']
            self.obs['ev_charging_power'] = self.ev_station.stats['current_charging_power']
            self.obs['ev_connected_count'] = self.ev_station.stats['connected_count']
            self.obs['ev_success_rate'] = self.ev_station.stats['current_success_rate']
            self.obs['avg_target_achieved'] = self.ev_station.stats['avg_target_achieved']
        else:
            self.obs['ev_connection_rate'] = 0.0
            self.obs['ev_charging_power'] = 0.0
            self.obs['ev_connected_count'] = 0.0
            self.obs['ev_success_rate'] = 0.0
            self.obs['avg_target_achieved'] = 0.0

        if self.wrap_observation:
            return self.wrap_obs(self.obs), info
        else:
            return self.obs, info

    def dss_step(self):
        """使用OpenDSS内置控制器进行控制（不控制电池），而不是接受外部动作。（已修改返回格式适配gymnasium的要求）
        Args:
        Returns:
            self.wrap_obs(self.obs): 环境的下一个观察值(array)
            reward: 执行动作获得的奖励(float)
            terminated: 回合是否因完成条件而终止(bool)
            truncated: 回合是否因时间限制等原因被截断(bool)
            info: 额外信息(dict)
        """
        assert self.circuit.dss_act, 'Env.circuit.dss_act must be True'

        # Note: 更新时间步
        prev_states = self.circuit.get_all_capacitor_statuses()
        prev_tapnums = self.circuit.get_all_regulator_tapnums()

        self.circuit.dss.ActiveCircuit.Solution.Solve()

        self.t += 1
        cap_statuses = self.circuit.get_all_capacitor_statuses()
        reg_statuses = self.circuit.get_all_regulator_tapnums()
        capdiff = np.array([abs(prev_states[c] - cap_statuses[c]) for c in prev_states])
        regdiff = np.array([abs(prev_tapnums[r] - reg_statuses[r]) for r in prev_tapnums])

        # OpenDSS does not control batteries
        soc_errs, dis_errs, bat_statuses = [], [], dict()

        # Note: 更新 obs
        bus_voltages = dict()
        for bus_name in self.all_bus_names:
            bus_voltages[bus_name] = self.circuit.bus_voltage(bus_name)
            bus_voltages[bus_name] = [bus_voltages[bus_name][i] for i in range(len(bus_voltages[bus_name])) if
                                      i % 2 == 0]

        self.obs['bus_voltages'] = bus_voltages
        self.obs['cap_statuses'] = cap_statuses
        self.obs['reg_statuses'] = reg_statuses
        self.obs['bat_statuses'] = bat_statuses
        self.obs['power_loss'] = - self.circuit.total_loss()[0] / self.circuit.total_power()[0]
        self.obs['time'] = self.t
        if self.observe_load:
            self.obs['load_profile_t'] = self.all_load_profiles.iloc[self.t % self.horizon].to_dict()

        # done = (self.t == self.horizon)
        # 为兼容gymnasium的更新要求: split 'done' into 'terminated' and 'truncated'
        terminated = (self.t == self.horizon)  # Episode is done due to termination condition
        truncated = False  # Not truncated due to time limit, etc.

        reward, info = self.reward_func.composite_reward(capdiff, regdiff, soc_errs[self.sto_num], dis_errs)
        # avoid dividing by zero
        # noinspection PyTypeChecker
        info.update({'av_cap_err': sum(capdiff) / (self.cap_num + 1e-10),
                     'av_reg_err': sum(regdiff) / (self.reg_num + 1e-10),
                     'av_dis_err': sum(dis_errs) / (self.bat_num + 1e-10),
                     'av_soc_err': sum(soc_errs) / (self.bat_num + 1e-10),
                     'av_soc': sum([soc for _, [soc, _] in bat_statuses.items()]) / (self.bat_num + 1e-10)})

        if self.wrap_observation:
            return self.wrap_obs(self.obs), reward, terminated, truncated, info
        else:
            return self.obs, reward, terminated, truncated, info

    def wrap_obs(self, obs):
        """打包观察字典到numpy.array Wrap the observation dictionary (i.e., self.obs) to a numpy array
        
        Attribute:
            obs: the observation dictionary generated at self.reset() and self.step()
        
        Return:
            a numpy array of observation.
        """
        key_obs = ['bus_voltages', 'cap_statuses', 'reg_statuses', 'bat_statuses']

        # 添加充电站实时指标
        ev_metrics = ['ev_connection_rate', 'ev_charging_power', 'ev_connected_count', 'ev_success_rate', 'avg_target_achieved']

        if self.observe_load:
            key_obs.append('load_profile_t')

        mod_obs = []
        for var_dict in key_obs:
            # node voltage is a dict of dict, we only take minimum phase node voltage
            # if var_dict == 'bus_voltages':
            #    for values in obs[var_dict].values():
            #        mod_obs.append(min(values))
            if var_dict in \
                    ['bus_voltages', 'cap_statuses', 'reg_statuses', 'bat_statuses', 'load_profile_t']:
                mod_obs = mod_obs + list(obs[var_dict].values())
            elif var_dict == 'power_loss':
                mod_obs.append(obs['power_loss'])

        # 添加充电站指标
        for metric in ev_metrics:
            mod_obs.append(obs[metric])

        return np.hstack(mod_obs)

    def build_graph(self):
        """建立一张 NetworkX 图用于后续使用
        
        Returns:
            Graph: Network graph
        """
        self.lines = dict()
        self.circuit.dss.ActiveCircuit.Lines.First  # noqa 将光标定位到第一个线路元素，不可带括号，但是pycharm有报告而我有强迫症
        while True:
            bus1 = self.circuit.dss.ActiveCircuit.Lines.Bus1.split('.', 1)[0].lower()
            bus2 = self.circuit.dss.ActiveCircuit.Lines.Bus2.split('.', 1)[0].lower()
            line_name = self.circuit.dss.ActiveCircuit.Lines.Name.lower()
            self.lines[line_name] = (bus1, bus2)
            if self.circuit.dss.ActiveCircuit.Lines.Next == 0:
                break

        transformer_names = self.circuit.dss.ActiveCircuit.Transformers.AllNames
        self.transformers = dict()
        for transformer_name in transformer_names:
            self.circuit.dss.ActiveCircuit.SetActiveElement('Transformer.' + transformer_name)
            buses = self.circuit.dss.ActiveCircuit.ActiveElement.BusNames
            # assert len(buses) == 2, 'Transformer {} has more than two terminals'.format(transformer_name)
            bus1 = buses[0].split('.', 1)[0].lower()
            bus2 = buses[1].split('.', 1)[0].lower()
            self.transformers[transformer_name] = (bus1, bus2)

        self.edges = [frozenset(edge) for _, edge in self.transformers.items()] + [frozenset(edge) for _, edge in
                                                                                   self.lines.items()]
        if len(self.edges) != len(set(self.edges)):
            print('There are ' + str(len(self.edges)) + ' edges and ' + str(
                len(set(self.edges))) + ' unique edges. Overlapping transformer edges')

        self.circuit.topology.add_edges_from(self.edges)
        # print(len(self.circuit.topology.nodes))
        # print(self.circuit.topology.nodes)
        # print(len(self.circuit.topology.edges))
        # print(self.circuit.topology.edges)

        # self.adj_mat = nx.adjacency_matrix(self.circuit.topology)
        # print(self.adj_mat.todense())
        return self.circuit.topology

    def plot_graph(self, node_bound='minimum',
                   vmin=0.95, vmax=1.05,
                   cmap='jet', figsize: tuple[float, float] = (18, 12),
                   text_loc_x=0, text_loc_y=400,
                   node_size=None, shift=None,
                   show_node_labels=None,
                   show_voltages=True,
                   show_controllers=True,
                   show_actions=False) -> tuple[plt.Figure, dict]:
        """用于绘制系统图，节点电压作为节点强度 Function to plot system graph with voltage as node intensity

        Args:
            node_bound (str): Determine to plot max/min node voltage for nodes with more than one phase
            vmin (float): Min heatmap intensity
            vmax (float): Max heatmap intensity
            cmap (str): Colormap
            figsize (tuple): Figure size
            text_loc_x (int): x-coordinate for timestamp
            text_loc_y (int): y-coordinate for timestamp
            node_size (int): Node size. If None, initialize with environment setting
            shift (int): shift of node label. If None, initialize with environment setting
            show_node_labels (bool): show node label. If None, initialize with environment setting
            show_voltages (bool): show voltages
            show_controllers (bool): show controllers
            show_actions (bool): show actions

        Returns:
            fig: Matplotlib figure
            pos: dictionary of node positions

        """
        node_size = self.node_size if node_size is None else node_size
        shift = self.shift if shift is None else shift
        show_node_labels = self.show_node_labels if show_node_labels is None else show_node_labels

        # get normalized node voltages
        voltages, nodes = [], []
        pos = dict()

        assert node_bound in ['maximum', 'minimum'], 'invalid node_bound'
        for busname in self.all_bus_names:
            self.circuit.dss.Circuits.SetActiveBus(busname)
            if not self.circuit.dss.Circuits.Buses.Coorddefined: continue
            x = self.circuit.dss.Circuits.Buses.x
            y = self.circuit.dss.Circuits.Buses.y

            pos[busname] = (x, y)
            nodes.append(busname)
            bus_volts = [self.circuit.dss.Circuits.Buses.puVmagAngle[i] for i in
                         range(len(self.circuit.dss.Circuits.Buses.puVmagAngle)) if i % 2 == 0]
            if node_bound == 'minimum':
                voltages.append(min(bus_volts))
            elif node_bound == 'maximum':
                voltages.append(max(bus_volts))

        fig = plt.figure(figsize=figsize)
        graph = nx.Graph()

        # local lines, transformers and edges
        HasLocation = lambda p: (p[0] in pos and p[1] in pos)
        loc_lines = [pair for pair in self.lines.values() if HasLocation(pair)]
        loc_trans = [pair for pair in self.transformers.values() if HasLocation(pair)]

        graph.add_edges_from(loc_lines + loc_trans)
        nx.draw_networkx_edges(graph, pos, loc_lines, edge_color='k', width=3, label='lines')
        nx.draw_networkx_edges(graph, pos, loc_trans, edge_color='r', width=3, label='transformers')
        if show_voltages:
            nx.draw_networkx_nodes(graph, pos, nodelist=nodes, node_color=voltages, vmin=vmin, vmax=vmax, cmap=cmap,
                                   node_size=node_size)
            # 创建 ScalarMappable 对象用于颜色条
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
            sm.set_array([])
            # 获取当前的 Axes 对象并添加颜色条
            ax = plt.gca()
            cbar = plt.colorbar(sm, ax=ax)
        else:
            nx.draw_networkx_nodes(graph, pos, nodelist=nodes, node_color=np.ones(len(voltages)), vmin=vmin, vmax=vmax,
                                   cmap=cmap, node_size=node_size)

        if show_node_labels:
            node_labels = {node: node for node in pos}
            nx.draw_networkx_labels(graph, pos, labels=node_labels, font_size=15)

        # show source bus
        loc = {self.source_bus: (pos[self.source_bus][0] + shift, pos[self.source_bus][1] - shift)}
        nx.draw_networkx_labels(graph, loc, labels={self.source_bus: 'src'}, font_size=15)

        if show_controllers:
            if self.cap_num > 0:
                labels = {self.circuit.capacitors[cap].bus1: 'cap' for cap in self.cap_names}
                labels = {k: v for k, v in labels.items() if k in pos}  # remove missing pos
                loc = {bus: (pos[bus][0] + shift, pos[bus][1] + shift) for bus in labels.keys()}
                nx.draw_networkx_labels(graph, loc, labels=labels, font_size=15,
                                        font_color='darkorange')
            if self.bat_num > 0:
                labels = {self.circuit.batteries[bat].bus1: 'bat' for bat in self.bat_names}
                labels = {k: v for k, v in labels.items() if k in pos}  # remove missing pos
                loc = {bus: (pos[bus][0] + shift, pos[bus][1] + shift) for bus in labels.keys()}
                nx.draw_networkx_labels(graph, loc, labels=labels, font_size=15,
                                        font_color='darkviolet')
            if self.reg_num > 0:
                regs = self.circuit.regulators
                labels = {(regs[r].bus1, regs[r].bus2): 'reg' for r in self.reg_names}
                # accept if one of the edge's node is in pos
                labels = {k: v for k, v in labels.items() if (k[0] in pos or k[1] in pos)}

                loc = dict()
                for key in labels.keys():
                    b1, b2 = key
                    lx, ly, count = 0.0, 0.0, 0
                    for b in list(key):
                        if b in pos:
                            ll = pos[b]
                            lx, ly, count = lx + ll[0], ly + ll[1], count + 1
                    lx, ly = lx / count, ly / count
                    loc[key] = (lx + shift, ly + shift)
                nx.draw_networkx_labels(graph, loc, labels=labels, font_size=15,
                                        font_color='darkred')

        if show_actions:
            plt.text(text_loc_x, text_loc_y, s='t=' + str(self.t) + ' Action: ' + self.str_action,
                     fontsize=18)
        elif show_voltages:
            plt.text(text_loc_x, text_loc_y, s='t=' + str(self.t), fontsize=18)

        return fig, pos

    def seed(self, seed):
        self.ActionSpace.seed(seed)

    def random_action(self):
        """随机采样动作
        
        Returns:
            Array: Random control actions
        """
        return self.ActionSpace.sample()

    def dummy_action_v0(self):
        """返回一个默认的零动作
    
        Returns:
            Array: Default zero actions
        """

        return [1] * self.cap_num + \
            [self.reg_act_num] * self.reg_num + \
            [0.0 if self.bat_act_num == np.inf else self.bat_act_num // 2] * self.bat_num

    def dummy_action(self):
        """返回一个默认的零动作。仅针对电池。
        
        Returns:
            Array: Default zero actions
        """
        if self.bat_act_num == np.inf:
            return np.zeros(self.bat_num)
        else:
            return [self.bat_act_num // 2] * self.bat_num

    def load_base_kW(self):
        """get base kW of load objects.

        see class Load in circuit.py for details on Load.feature

        Returns:
            base kW of Load objects.
        """
        basekW = dict()
        for load in self.circuit.loads.keys():
            basekW[load[5:]] = self.circuit.loads[load].feature[1]  # learn: 这个feature的获取方式
        return basekW

    def step_v1(self, action: np.ndarray):
        """Steps through one step of environment and calls DSS solver after one control action. 适配gymnasium的更新要求。

        Args:
            action (nd.array): Array of actions for capacitors, regulators and batteries

        Returns:
            self.wrap_obs(self.obs): 环境的下一个观察值(array)
            reward: 执行动作获得的奖励(float)
            terminated: 回合是否因完成条件而终止(bool)
            truncated: 回合是否因时间限制等原因被截断(bool)
            info: 额外信息(dict)
        """
        action_idx = 0
        self.str_action = ''  # the action string to be printed at self.plot_graph()

        # 控制 capacitor
        if self.cap_num > 0:
            statuses = action[action_idx:action_idx + self.cap_num]
            capdiff = self.circuit.set_all_capacitor_statuses(statuses)
            cap_statuses = {cap: status for cap, status in zip(self.circuit.capacitors.keys(), statuses)}
            action_idx += self.cap_num
            self.str_action += 'Cap Status:' + str(statuses)
        else:
            capdiff, cap_statuses = [], dict()

        # 控制 regulator
        if self.reg_num > 0:
            tapnums = action[action_idx:action_idx + self.reg_num]
            regdiff = self.circuit.set_all_regulator_tappings(tapnums)
            reg_statuses = {reg: self.circuit.regulators[reg].tap for reg in self.reg_names}
            action_idx += self.reg_num
            self.str_action += 'Reg Tap Status:' + str(tapnums)
        else:
            regdiff, reg_statuses = [], dict()

        # 控制 battery
        if self.bat_num > 0:
            states = action[action_idx:]
            self.circuit.set_all_batteries_before_solve(states)
            self.str_action += 'Bat Status:' + str(states)

        self.circuit.dss.ActiveCircuit.Solution.Solve()

        # 更新电池信息 kWh. record soc_err and discharge_err
        if self.bat_num > 0:
            soc_errs, dis_errs = self.circuit.set_all_batteries_after_solve()
            bat_statuses = {name: [bat.soc, -1 * bat.actual_power() / bat.max_kw] for name, bat in
                            self.circuit.batteries.items()}
        else:
            soc_errs, dis_errs, bat_statuses = [], [], dict()

        # 更新时间步
        self.t += 1

        # 更新 obs
        bus_voltages = dict()
        for bus_name in self.all_bus_names:
            bus_voltages[bus_name] = self.circuit.bus_voltage(bus_name)
            bus_voltages[bus_name] = [bus_voltages[bus_name][i] for i in range(len(bus_voltages[bus_name])) if
                                      i % 2 == 0]

        self.obs['bus_voltages'] = bus_voltages
        self.obs['cap_statuses'] = cap_statuses
        self.obs['reg_statuses'] = reg_statuses
        self.obs['bat_statuses'] = bat_statuses
        self.obs['power_loss'] = - self.circuit.total_loss()[0] / self.circuit.total_power()[0]
        self.obs['time'] = self.t
        if self.observe_load:
            self.obs['load_profile_t'] = self.all_load_profiles.iloc[self.t % self.horizon].to_dict()  # 截取负载曲线包括在obs中

        # 为兼容gymnasium的更新要求: split 'done' into 'terminated' and 'truncated'
        terminated = (self.t == self.horizon)  # Episode is done due to termination condition
        truncated = False  # Not truncated due to time limit, etc.

        reward, info = self.reward_func.composite_reward(capdiff, regdiff, soc_errs, dis_errs)

        # noinspection PyTypeChecker
        info.update({'av_cap_err': sum(capdiff) / (self.cap_num + 1e-10),
                     'av_reg_err': sum(regdiff) / (self.reg_num + 1e-10),
                     'av_dis_err': sum(dis_errs) / (self.bat_num + 1e-10),
                     'av_soc_err': sum(soc_errs) / (self.bat_num + 1e-10),
                     'av_soc': sum([soc for _, [soc, _] in bat_statuses.items()]) / (
                             self.bat_num + 1e-10)})  # avoid dividing by zero

        if self.wrap_observation:
            return self.wrap_obs(self.obs), reward, terminated, truncated, info
        else:
            return self.obs, reward, terminated, truncated, info

    def reset_v0(self, load_profile_idx=0):
        """重置环境以开启新的回合
        Args:
            load_profile_idx: 负载配置文件的ID号（默认: 0）
        Returns:
            observation: 封装后的观察值
        """
        # reset time
        self.t = 0

        # choose load profile
        self.load_profile.choose_loadprofile(load_profile_idx)
        self.all_load_profiles = self.load_profile.get_loadprofile(load_profile_idx)

        # re-compile dss and reset batteries
        self.circuit.reset()

        # node voltages
        bus_voltages = dict()
        for bus_name in self.all_bus_names:
            bus_voltages[bus_name] = self.circuit.bus_voltage(bus_name)
            bus_voltages[bus_name] = [bus_voltages[bus_name][i] for i in range(len(bus_voltages[bus_name])) if
                                      i % 2 == 0]
        self.obs['bus_voltages'] = bus_voltages

        # status of capacitor
        cap_statuses = {name: cap.status for name, cap in self.circuit.capacitors.items()}
        self.obs['cap_statuses'] = cap_statuses

        # status of regulator
        reg_statuses = {name: reg.tap for name, reg in self.circuit.regulators.items()}
        self.obs['reg_statuses'] = reg_statuses

        # status of battery
        bat_statuses = {name: [bat.soc, -1 * bat.actual_power() / bat.max_kw] for name, bat in
                        self.circuit.batteries.items()}
        self.obs['bat_statuses'] = bat_statuses

        # total power loss
        self.obs['power_loss'] = -self.circuit.total_loss()[0] / self.circuit.total_power()[0]

        # time step tracker
        self.obs['time'] = self.t

        # load for current timestep
        if self.observe_load:
            self.obs['load_profile_t'] = self.all_load_profiles.iloc[self.t].to_dict()

        # Edge weight
        # elf.obs['Y_matrix'] = self.circuit.edge_weight

        if self.wrap_observation:  # 初始化函数内已直接设置为真
            return self.wrap_obs(self.obs)
        else:
            return self.obs
