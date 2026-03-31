# -*- coding: utf-8 -*-
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-DSSGym-Commercial
# Copyright (c) 2025 Su Chenhui
# @File: ev_bms_v01.py
# @Time: 2025/5/8 22:04
# @Author: Gan Mocai
# @Software: PyCharm, VS Code

"""
更新的EVBMS，与旧EVBMS共存。

Improve:
    1. 改进了计算逻辑：
        先根据电池的最大支持功率、协议类型和SOC计算出EV的功率需求值；
        然后将这个功率需求值与充电桩提供的功率做比较，取较小值作为返回值。
    2. 添加了开关功能：
        可以选择是否启用功率需求计算逻辑。
        当不启用时，直接返回充电桩的功率值而不进行需求计算。
Note:
    和v00存在同样问题，实际不仅返回需求值给充电桩，也模拟了充电桩的输出行为（假设充电桩会尽量返回容量内的需求值）。
"""

#%% 导入区
from typing import List, Dict, Any, Optional, Union, Tuple

#%% 预设区

"""
下面是禁用print的代码，根据Claude3.7，其作用范围：
    1. 该文件不会影响重定义之前执行的print代码【当函数被定义时，Python不会立即确定函数内的print指向哪个对象，而是在函数被调用执行时才解析print的引用。即使函数定义在print重定义之前，只要函数调用在重定义之后，该函数内的print就会使用重定义后的版本。】
    2. 其他文件导入了这个模块但没有显式导入 print 函数，其他文件中的 print 调用不会受到影响
    3. 如果其他文件使用 from ev_bms_v01 import *，可能会导入重定义的 print 函数

当主程序导入EVBMS后：
    1. 主程序中的print调用不会被禁用，仍会正常输出
    2. 但是EVBMS类内部的所有print调用会被禁用，因为：
        2.1 类定义时已在模块作用域内重定义了print
        2.2 类方法调用时会使用模块作用域内的print定义
        2.3 即使是在主程序中实例化的EVBMS对象，其方法内的print也会被禁用
"""

# 保存原始print函数
original_print = print
# 重定义为无操作函数
print = lambda *args, **kwargs: None


#%% 正文区

class EVBMS:
    """
    电动汽车电池管理系统 (EVBMS)
    根据当前SOC、电池支持的最大充电功率、充电桩容量和充电协议，计算并控制充电功率.
    其核心函数只有两个，set_soc, calculate_charge_power，is_charging状态不影响后者计算并返回一个值。
    """

    # 充电协议类型定义
    PROTOCOL_MCC = 0  # 多段恒流充电 (对应curve_type=0)
    PROTOCOL_Decay = 1  # 上升-平台-衰减 (对应curve_type=1)
    PROTOCOL_CC_CV = 2  # 恒功率充电 (对应curve_type=2)

    def __init__(self,
                 battery_capacity: float = 60.0,
                 max_battery_charge_power: float = 60.0,
                 initial_soc: float = 0.2,
                 charge_protocol: int = 0,
                 enable_power_demand: bool = True):
        """
        初始化EVBMS

        参数:
            battery_capacity: 电池容量(kWh)
            max_battery_charge_power: 电池支持的最大充电功率(kW)
            initial_soc: 初始SOC百分比(0-1)
            charge_protocol: 充电协议类型(0=多段恒流, 1=上升-平台-衰减, 2=恒功率)
            enable_power_demand: 是否启用功率需求计算逻辑
        """
        # 基础参数
        self.battery_capacity = battery_capacity  # 电池容量(kWh)
        self.max_battery_charge_power = max_battery_charge_power  # 最大充电功率(kW)
        self.current_soc = initial_soc  # 当前电量百分比(0-1)
        self.charge_protocol = charge_protocol  # 充电协议
        self.enable_power_demand = enable_power_demand  # 是否启用功率需求计算

        self.charger_power = None

        # 充电历史记录
        self.charging_history = []
        self.time_segment_counter = 0  # 时间段序号计数器

        # 环境因素
        self.battery_temperature = 25.0  # 默认电池温度25°C

        # 内部状态
        self.current_charge_power = 0.0  # 当前充电功率
        self.power_demand = 0.0  # EV功率需求值
        self.is_charging = False  # 是否正在充电
        self.charging_duration = 0  # 充电持续时间(秒)

    def start_charging(self, charger_power: float) -> float:
        """
        开始充电
        Note: 这里可能会重新设置charger_power

        参数:
            charger_power: 充电桩提供的最大功率(kW)

        返回:
            实际充电功率(kW)
        """
        if self.is_charging:
            print("已经在充电中")
            return self.current_charge_power

        self.is_charging = True
        self.charging_duration = 0
        if self.charger_power is None:
            self.update_charger_power(charger_power)

        # 计算初始充电功率
        power = self.calculate_charge_power(charger_power)
        self.set_charge_power(power)

        return self.current_charge_power

    def stop_charging(self) -> Optional[Dict[str, Any]]:
        """
        停止充电

        返回:
            充电会话统计
        """
        if not self.is_charging:
            print("没有正在进行的充电")
            return None

        start_soc = self.charging_history[0]['soc'] if self.charging_history else self.current_soc

        charging_session = {
            "duration": self.charging_duration,
            "start_soc": start_soc,
            "end_soc": self.current_soc,
            "total_segments": self.time_segment_counter,
            "power_history": self.charging_history.copy()
        }

        self.is_charging = False
        self.current_charge_power = 0.0
        self.power_demand = 0.0
        self.charging_history = []
        self.time_segment_counter = 0  # 重置时间段计数器

        return charging_session

    def update_charging_status(self, charger_power: float, elapsed_seconds: float) -> float:
        """
        更新充电状态

        参数:
            charger_power: 充电桩当前提供的最大功率(kW)
            elapsed_seconds: 已经充电的时间(秒)

        返回:
            更新后的充电功率(kW)
        """
        if not self.is_charging:
            print("没有正在进行的充电")
            return 0.0

        # 更新充电时间
        self.charging_duration = elapsed_seconds

        # 模拟SOC增加 (简化模型)
        # 假设在当前功率下，每小时充入的电量相当于电池容量的 currentPower/batteryCapacity 比例
        hours_fraction = elapsed_seconds / 3600
        energy_charged = self.current_charge_power * hours_fraction
        soc_increase = (energy_charged / self.battery_capacity)  # SOC增加量(0-1)

        # 更新SOC (不考虑充电效率损失，实际应该有系数)
        self.current_soc = min(1.0, self.current_soc + soc_increase)

        # 根据新SOC重新计算充电功率
        new_power = self.calculate_charge_power(charger_power)
        self.set_charge_power(new_power)

        return self.current_charge_power

    def set_charge_power(self, power: float) -> None:
        """
        直接设置充电功率，绕开BMS的处理

        参数:
            power: 充电功率(kW)
        """
        self.current_charge_power = power

        # 递增时间段序号
        self.time_segment_counter += 1

        # 记录到历史
        self.charging_history.append({
            "segment": self.time_segment_counter,
            "soc": self.current_soc,
            "power": power,
            "temperature": self.battery_temperature,
            "power_demand": self.power_demand  # 添加功率需求记录
        })

    def calculate_power_demand(self) -> float:
        """
        计算EV功率需求值，基于电池支持的最大充电功率、充电协议和当前SOC

        返回:
            计算后的功率需求(kW)
        """
        # 初始功率需求为电池支持的最大充电功率
        power_demand = self.max_battery_charge_power

        # 根据充电协议和SOC调整功率需求
        if self.charge_protocol == self.PROTOCOL_MCC:  # 多段恒流充电
            if self.current_soc < 0.3:
                # 第一阶段: 最大功率充电
                pass  # 保持当前功率不变
            elif self.current_soc < 0.5:
                # 第二阶段: 80%功率
                power_demand = power_demand * 0.8
            elif self.current_soc < 0.7:
                # 第三阶段: 60%功率
                power_demand = power_demand * 0.6
            elif self.current_soc < 0.85:
                # 第四阶段: 40%功率
                power_demand = power_demand * 0.4
            else:
                # 最后阶段: 25%功率
                power_demand = power_demand * 0.25

        elif self.charge_protocol == self.PROTOCOL_Decay:  # 上升-平台-衰减
            if self.current_soc < 0.3:
                # 上升阶段: 功率随SOC线性增加
                power_demand = power_demand * (0.5 + self.current_soc / 0.6)  # 从50%额定功率开始线性增加
            elif self.current_soc < 0.7:
                # 平台阶段: 保持最大功率
                pass  # 保持当前功率不变
            else:
                # 衰减阶段: 功率随SOC增加而双曲线下降
                k = 0.15  # 衰减系数
                reduce_factor = 1 / (1 + k * (self.current_soc - 0.7))
                power_demand = power_demand * reduce_factor

        elif self.charge_protocol == self.PROTOCOL_CC_CV:  # 恒功率充电
            # 恒功率模式下功率不变，但在SOC很高时略微降低以保护电池
            if self.current_soc > 0.9:
                power_demand = power_demand * 0.9  # 轻微降低功率

        else:  # 未知充电协议，默认退化到多段恒流模式
            if self.current_soc < 0.3:
                pass  # 保持当前功率不变
            elif self.current_soc < 0.5:
                power_demand = power_demand * 0.8
            elif self.current_soc < 0.7:
                power_demand = power_demand * 0.6
            elif self.current_soc < 0.85:
                power_demand = power_demand * 0.4
            else:
                power_demand = power_demand * 0.25

        # 考虑温度影响 (简化模型)
        if self.battery_temperature < 0:
            # 低温时减少充电功率
            power_demand = power_demand * 0.5
        elif self.battery_temperature > 40:
            # 高温时减少充电功率
            power_demand = power_demand * 0.7

        # 限制最小充电功率为1kW
        return max(1.0, power_demand)

    def update_charger_power(self, charger_power: float) -> float:
        """
        接收并更新充电桩功率
        
        参数:
            charger_power: 充电桩提供的功率容量(kW)
            
        返回:
            更新后的充电桩功率(kW)
        """
        if self.charger_power is None:
            self.charger_power = charger_power
            print(f"设置{self.charger_power=}成功。")
        elif self.charger_power <= 10:  # BMS charger_power <= 10 可重设
            print(f"历史设置{self.charger_power=}已更新为{charger_power}。")
            self.charger_power = charger_power
        else:
            print(f'历史设置{self.charger_power=}，无法更新为{charger_power=}。')
            charger_power = self.charger_power
        
        return charger_power

    def calculate_charge_power(self, charger_power:float=None, enable_power_demand:bool=None) -> float:
        """
        计算合适的充电功率，并允许在此处更新是否实际启用BMS计算。

        参数:
            charger_power: 充电桩提供的功率容量(kW)
            enable_power_demand: 要更新为的开关状态

        返回:
            计算后的充电功率(kW)
        """
        if enable_power_demand is not None:
            self.enable_power_demand = enable_power_demand

        # 判断是否启用功率需求计算
        if self.enable_power_demand is False:
            # 不启用功率需求计算，直接返回充电桩功率
            self.power_demand = 0.0  # 清空功率需求值
            print(f"功率需求已禁用，直接使用桩侧设置功率: {charger_power:.2f} kW。")
            return charger_power

        # 更新充电桩功率
        if charger_power is not None:
            charger_power = self.update_charger_power(charger_power)

        # 计算EV的功率需求值
        self.power_demand = self.calculate_power_demand()

        # 比较功率需求与充电桩功率，取较小值
        power = min(charger_power, self.power_demand)

        print(f"功率需求: {self.power_demand:.2f} kW, 充电桩功率: {charger_power:.2f} kW, 最终充电功率: {power:.2f} kW。")

        return power

    def set_battery_temperature(self, temperature: float) -> None:
        """
        设置电池温度

        参数:
            temperature: 电池温度(°C)
        """
        self.battery_temperature = temperature

        # 如果正在充电，重新计算功率
        if self.is_charging:
            new_power = self.calculate_charge_power(self.charger_power)
            self.set_charge_power(new_power)

    def get_charging_history(self) -> List[Dict[str, Any]]:
        """
        获取充电历史记录

        返回:
            充电历史记录
        """
        return self.charging_history

    def get_charging_status(self) -> Dict[str, Any]:
        """
        获取当前充电状态

        返回:
            充电状态
        """
        # 协议名称映射
        protocol_names = {
            self.PROTOCOL_MCC: "多段恒流充电",
            self.PROTOCOL_Decay: "上升-平台-衰减",
            self.PROTOCOL_CC_CV: "恒功率充电"
        }

        return {
            "is_charging": self.is_charging,
            "current_soc": self.current_soc,
            "current_power": self.current_charge_power,
            "power_demand": self.power_demand,  # 添加功率需求
            "power_demand_enabled": self.enable_power_demand,  # 添加功率需求计算启用状态
            "protocol": self.charge_protocol,
            "protocol_name": protocol_names.get(self.charge_protocol, "Unknown"),
            "duration": self.charging_duration,
            "temperature": self.battery_temperature
        }

    def set_soc(self, new_soc: float) -> bool:
        """
        直接设置电池SOC值

        参数:
            new_soc: 新的SOC值(0-1)

        返回:
            操作是否成功
        """
        # 验证SOC值是否在有效范围内
        if not (0 <= new_soc <= 1):
            print(f"错误: SOC值 {new_soc} 超出有效范围(0-1)。")
            return False

        # 更新SOC
        old_soc = self.current_soc
        self.current_soc = new_soc

        # 如果正在充电，根据新SOC重新计算充电功率
        if self.is_charging:
            # 计算功率前获取当前充电桩功率上限
            charger_power = self.charger_power
            new_power = self.calculate_charge_power(charger_power)
            self.set_charge_power(new_power)
            print(f"SOC从 {old_soc:.2f} 更新为 {new_soc:.2f}, 充电功率调整为 {new_power:.2f} kW。")
        else:
            print(f"SOC从 {old_soc:.2f} 更新为 {new_soc:.2f}。")

        return True


# 使用示例
if __name__ == "__main__":
    # 创建BMS实例 - 启用功率需求计算
    bms = EVBMS(
        battery_capacity=60.0,  # 电池容量60kWh
        max_battery_charge_power=120.0,  # 电池最大接受充电功率120kW
        initial_soc=0.2,  # 初始SOC 0.2 (20%)
        charge_protocol=EVBMS.PROTOCOL_MCC,  # 充电协议: 多段恒流 (0)
        enable_power_demand=True  # 启用功率需求计算
    )

    # 开始充电，充电桩提供60kW功率
    print(f"开始充电, 功率: {bms.start_charging(60.0):.2f} kW")
    print(f"当前状态: {bms.get_charging_status()}")

    # 模拟充电10分钟
    print(f"\n充电10分钟后, 功率: {bms.update_charging_status(60.0, 600):.2f} kW")
    print(f"当前SOC: {bms.current_soc:.2f}")
    print(f"功率需求: {bms.power_demand:.2f} kW")

    # 模拟充电1小时
    print(f"\n充电1小时后, 功率: {bms.update_charging_status(60.0, 3600):.2f} kW")
    print(f"当前SOC: {bms.current_soc:.2f}")
    print(f"功率需求: {bms.power_demand:.2f} kW")

    # 模拟电池温度变化
    bms.set_battery_temperature(5.0)
    print(f"\n电池温度降至5°C, 功率调整为: {bms.current_charge_power:.2f} kW")
    print(f"功率需求: {bms.power_demand:.2f} kW")

    # 测试不同充电桩功率下的充电
    print("\n测试不同充电桩功率:")
    # 充电桩功率高于需求
    bms.set_soc(0.4)  # 设置SOC到需要80%最大功率的区间
    print(f"SOC: {bms.current_soc:.2f}")
    bms.calculate_charge_power(150.0)  # 充电桩功率150kW，高于需求
    print(f"充电桩150kW (高于需求), 功率: {bms.current_charge_power:.2f} kW")

    # 充电桩功率低于需求
    bms.calculate_charge_power(40.0)  # 充电桩功率40kW，低于需求
    print(f"充电桩40kW (低于需求), 功率: {bms.current_charge_power:.2f} kW")

    # 创建使用不同充电协议的BMS实例
    print("\n测试不同充电协议:")

    # 上升-平台-衰减
    decay_bms = EVBMS(
        battery_capacity=60.0,
        max_battery_charge_power=120.0,
        initial_soc=0.2,  # 从0.2开始，观察上升阶段
        charge_protocol=EVBMS.PROTOCOL_Decay  # 上升-平台-衰减 (1)
    )
    decay_bms.start_charging(60.0)
    print(f"上升-平台-衰减协议(0.2 SOC)功率: {decay_bms.current_charge_power:.2f} kW")
    print(f"功率需求: {decay_bms.power_demand:.2f} kW")

    # 让SOC增加到0.5，观察平台阶段
    decay_bms.set_soc(0.5)
    print(f"上升-平台-衰减协议(0.5 SOC)功率: {decay_bms.current_charge_power:.2f} kW")
    print(f"功率需求: {decay_bms.power_demand:.2f} kW")

    # 让SOC增加到0.85，观察衰减阶段
    decay_bms.set_soc(0.85)
    print(f"上升-平台-衰减协议(0.85 SOC)功率: {decay_bms.current_charge_power:.2f} kW")
    print(f"功率需求: {decay_bms.power_demand:.2f} kW")

    # 恒功率充电
    constant_bms = EVBMS(
        battery_capacity=60.0,
        max_battery_charge_power=120.0,
        initial_soc=0.5,
        charge_protocol=EVBMS.PROTOCOL_CC_CV  # 恒功率充电 (2)
    )
    constant_bms.start_charging(60.0)
    print(f"恒功率协议(0.5 SOC)功率: {constant_bms.current_charge_power:.2f} kW")
    print(f"功率需求: {constant_bms.power_demand:.2f} kW")

    # 让SOC增加到0.95，观察高SOC时的行为
    constant_bms.set_soc(0.95)
    print(f"恒功率协议(0.95 SOC)功率: {constant_bms.current_charge_power:.2f} kW")
    print(f"功率需求: {constant_bms.power_demand:.2f} kW")

    # 继续使用原BMS充电直到SOC达到0.85
    print("\n继续多段恒流充电:")
    while bms.current_soc < 0.85:
        bms.update_charging_status(60.0, 600)  # 每次充电10分钟

    print(f"充电至0.85 SOC, 当前功率: {bms.current_charge_power:.2f} kW")
    print(f"功率需求: {bms.power_demand:.2f} kW")

    # 测试直接设置SOC
    print("\n测试直接设置SOC:")
    # 设置有效的SOC
    result = bms.set_soc(0.9)
    print(f"设置SOC为0.9: {'成功' if result else '失败'}")
    print(f"当前功率: {bms.current_charge_power:.2f} kW")
    print(f"功率需求: {bms.power_demand:.2f} kW")

    # 设置无效的SOC
    result = bms.set_soc(1.2)
    print(f"设置SOC为1.2: {'成功' if result else '失败'}")
    print(f"当前SOC: {bms.current_soc:.2f}")

    # 打印历史记录，查看时间段序号和功率需求
    print("\n充电历史记录:")
    for record in bms.get_charging_history():
        print(
            f"时间段: {record['segment']}, SOC: {record['soc']:.2f}, 功率: {record['power']:.2f} kW, 功率需求: {record.get('power_demand', 'N/A')} kW")

    # 停止充电并查看充电会话
    session = bms.stop_charging()
    print(f"\n充电会话总结:")
    print(f"持续时间: {session['duration'] / 60:.2f} 分钟")
    print(f"SOC变化: {session['start_soc']:.2f} -> {session['end_soc']:.2f}")
    print(f"时间段总数: {session['total_segments']}")
    print(f"功率历史记录数: {len(session['power_history'])}")

    # 测试禁用功率需求计算
    print("\n测试禁用功率需求计算:")
    # 创建禁用功率需求计算的BMS实例
    disabled_bms = EVBMS(
        battery_capacity=60.0,
        max_battery_charge_power=120.0,
        initial_soc=0.5,
        charge_protocol=EVBMS.PROTOCOL_MCC,
        enable_power_demand=False  # 禁用功率需求计算
    )

    # 开始充电，充电桩提供80kW功率
    disabled_bms.start_charging(80.0)
    print(f"禁用功率需求计算, 充电桩功率80kW, 实际充电功率: {disabled_bms.current_charge_power:.2f} kW")
    print(f"功率需求值: {disabled_bms.power_demand}")

    # 修改SOC
    disabled_bms.set_soc(0.7)
    print(f"SOC调整为0.7后, 实际充电功率: {disabled_bms.current_charge_power:.2f} kW")

    # 获取状态
    status = disabled_bms.get_charging_status()
    print(f"充电状态: {status}")

    # 启用功率需求计算
    print("\n重新启用功率需求计算:")
    disabled_bms.enable_power_demand = True
    # 重新计算充电功率
    new_power = disabled_bms.calculate_charge_power(80.0)
    disabled_bms.set_charge_power(new_power)
    print(f"启用功率需求计算后, 实际充电功率: {disabled_bms.current_charge_power:.2f} kW")
    print(f"功率需求值: {disabled_bms.power_demand:.2f} kW")
