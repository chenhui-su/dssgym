# -*- coding: utf-8 -*-
# Copyright 2025 Su Chenhui
# @CreateTime : 2025-07-06 14:09
# @Author : Su Chenhui
# @FIle : ev_model_v00.py
# @Software : PyCharm

"""
专门的 EV 类

Improve:
    1. 从 `circuit.py` 中添加的 EV 操作中分离构建

Note:

"""

from ev_bms_v01 import EVBMS


class EV:
    """
    电动汽车类，包含电池特性、充电需求和状态信息

    Attributes:
        name (str): 电动汽车唯一标识
        battery_capacity (float): 电池容量(kWh)
        max_charge_power (float): 最大充电功率(kW)
        initial_soc (float): 初始SOC(0-1)
        target_soc (float): 目标SOC(0-1)
        arrival_time (int): 到达时间步
        departure_time (int): 离开时间步
        charge_protocol (int): 充电协议类型
        current_soc (float): 当前SOC(0-1)
        connection_status (bool): 是否已连接到充电桩
        connection_point (int): 连接的充电桩ID
        charging_power (float): 当前充电功率(kW)
        bms (EVBMS): 电池管理系统对象
        charge_history (list): 充电历史记录
    """

    def __init__(self, name, battery_capacity, max_charge_power,
                 initial_soc, target_soc, arrival_time, departure_time,
                 charge_protocol=0, pf=-0.98):
        """
        初始化电动汽车对象

        Args:
            name (str): 电动汽车唯一标识
            battery_capacity (float): 电池容量(kWh)
            max_charge_power (float): 最大充电功率(kW)
            initial_soc (float): 初始SOC(0-1)
            target_soc (float): 目标SOC(0-1)
            arrival_time (int): 到达时间步
            departure_time (int): 离开时间步
            charge_protocol (int): 充电协议类型，默认为0(多段恒功率)
            pf (float): 功率因数，默认为-0.98
        """
        # 基本参数
        self.name = name
        self.battery_capacity = battery_capacity
        self.max_charge_power = max_charge_power
        self.initial_soc = initial_soc
        self.target_soc = target_soc
        self.arrival_time = arrival_time
        self.departure_time = departure_time
        self.charge_protocol = charge_protocol
        self.pf = pf

        # 运行时状态
        self.current_soc = initial_soc
        self.connection_status = False
        self.connection_point = None
        self.charging_power = 0.0
        self.energy_charged = 0.0  # 累计充电能量(kWh)
        self.waiting_since = None  # 开始等待的时间步

        # 创建BMS对象
        self.bms = EVBMS(
            battery_capacity=battery_capacity,
            max_battery_charge_power=max_charge_power,
            initial_soc=initial_soc,
            charge_protocol=charge_protocol
        )

        # 充电历史
        self.charge_history = []

    def connect(self, connection_point):
        """连接到充电桩"""
        self.connection_status = True
        self.connection_point = connection_point
        return True

    def disconnect(self):
        """断开与充电桩的连接"""
        self.connection_status = False
        self.connection_point = None
        self.charging_power = 0.0
        return True

    def start_waiting(self, time_step):
        """开始等待充电"""
        self.waiting_since = time_step

    def calculate_charge_power(self, available_power):
        """
        计算实际充电功率

        Args:
            available_power (float): 充电桩可提供的最大功率(kW)

        Returns:
            float: 实际充电功率(kW)
        """
        if not self.connection_status:
            return 0.0

        # 使用BMS计算实际充电功率
        power = self.bms.calculate_charge_power(min(available_power, self.max_charge_power))
        self.charging_power = power
        return power

    def update_soc(self, time_step, duration=0.25):
        """
        更新SOC和充电状态

        Args:
            time_step (int): 当前时间步
            duration (float): 时间步长度(小时)，默认为0.25(15分钟)

        Returns:
            dict: 更新后的状态信息
        """
        if not self.connection_status or self.charging_power <= 0:
            return {
                'time_step': time_step,
                'soc': self.current_soc,
                'power': 0,
                'energy_charged': 0
            }

        # 计算充电能量
        energy_charged = self.charging_power * duration
        self.energy_charged += energy_charged

        # 更新SOC
        soc_increase = energy_charged / self.battery_capacity
        prev_soc = self.current_soc
        self.current_soc = min(1.0, self.current_soc + soc_increase)

        # 更新BMS中的SOC
        self.bms.set_soc(self.current_soc * 100)  # BMS需要百分比形式的SOC

        # 记录充电历史
        charge_record = {
            'time_step': time_step,
            'soc': self.current_soc,
            'power': self.charging_power,
            'energy_charged': energy_charged,
            'soc_increase': self.current_soc - prev_soc
        }
        self.charge_history.append(charge_record)

        return charge_record

    def is_target_reached(self):
        """检查是否达到目标SOC"""
        return self.current_soc >= self.target_soc

    def get_satisfaction_ratio(self):
        """计算充电满足率"""
        if self.target_soc <= self.initial_soc:
            return 1.0

        actual_increase = self.current_soc - self.initial_soc
        target_increase = self.target_soc - self.initial_soc

        return min(1.0, actual_increase / target_increase)

    def get_remaining_charge_time(self, charge_power):
        """
        估计剩余充电时间(小时)

        Args:
            charge_power (float): 假设的充电功率(kW)

        Returns:
            float: 估计的剩余充电时间(小时)
        """
        if charge_power <= 0 or self.current_soc >= self.target_soc:
            return 0

        remaining_energy = (self.target_soc - self.current_soc) * self.battery_capacity
        return remaining_energy / charge_power

    def to_dict(self):
        """返回EV状态的字典表示"""
        return {
            'name': self.name,
            'capacity': self.battery_capacity,
            'max_power': self.max_charge_power,
            'initial_soc': self.initial_soc,
            'target_soc': self.target_soc,
            'current_soc': self.current_soc,
            'arrival_time': self.arrival_time,
            'departure_time': self.departure_time,
            'connection_status': self.connection_status,
            'connection_point': self.connection_point,
            'charging_power': self.charging_power,
            'energy_charged': self.energy_charged,
            'satisfaction_ratio': self.get_satisfaction_ratio()
        }

    def __repr__(self):
        return f"EV({self.name}, SOC: {self.current_soc:.2f}, Power: {self.charging_power:.1f}kW)"
