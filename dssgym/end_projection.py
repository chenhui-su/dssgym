# -*- coding: utf-8 -*-
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-DSSGym-Commercial
# Copyright (c) 2025 Su Chenhui
# @File: end_projection.py
# @Time: 2025/5/9 15:07
# @Author: Gan Mocai
# @Software: PyCharm, VS Code

"""
为适应不对称空间出现的连续动作空间收缩（导致储能不放电影响），添加动作后处理映射。
对于离散动作空间类似处理。

Note: 针对DSSGym，不具有泛用性。
"""

import gymnasium as gym
import numpy as np


class CustomActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.sto_num = env.sto_num
        # 根据动作空间类型确定获取bat_num的方式
        self.is_discrete = isinstance(env.action_space, gym.spaces.MultiDiscrete)
        self.bat_num = env.action_space.shape[0] if hasattr(env.action_space, 'shape') else len(env.action_space.nvec)

        if self.is_discrete:
            # 如果原始空间是离散的，获取动作空间基数
            self.original_nvec = env.action_space.nvec
            self.storage_act_num = self.original_nvec[0] if self.sto_num > 0 else 0
            self.ev_act_num = self.original_nvec[self.sto_num] if self.sto_num < self.bat_num else 0

            # 创建统一的离散动作空间
            self.action_space = gym.spaces.MultiDiscrete([self.storage_act_num] * self.bat_num)
        else:
            # 连续动作空间，统一为[0,1]区间
            self.action_space = gym.spaces.Box(
                low=0.0, high=1.0,
                shape=(self.bat_num,),
                dtype=np.float32
            )

    def action(self, action):
        """将统一范围的动作映射到原始的不均匀范围"""
        if self.is_discrete:
            # 处理离散动作空间
            mapped_action = np.array(action, dtype=int)
            # 储能设备保持原有动作范围
            mapped_action[:self.sto_num] = action[:self.sto_num]
            # 非储能设备仅使用前半部分动作范围（不放电）
            mapped_action[self.sto_num:] = action[self.sto_num:] // 2
            return mapped_action
        else:
            # 处理连续动作空间
            mapped_action = np.zeros_like(action, dtype=np.float32)
            # 前sto_num个动作映射到[-1,1]
            mapped_action[:self.sto_num] = 2.0 * action[:self.sto_num] - 1.0
            # 剩余动作映射到[-1,0]（仅充电，不放电）
            mapped_action[self.sto_num:] = action[self.sto_num:] * (-1.0)
            return mapped_action

