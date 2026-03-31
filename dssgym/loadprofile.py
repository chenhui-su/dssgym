# Copyright 2025 Su Chenhui
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-DSSGym-Commercial

"""
Todo:
    1. 处理这里的一个冗余逻辑，检测到dss文件中的负载曲线不带daily后，自动生成新的dss文件.
    2. 负载曲线生成的参考源添加自定义机制，不局限于

`loadprofile.py` 模块用于为电力系统中的负载生成负载曲线。
它从指定的文件夹读取负载形状文件，并根据指定的步数和比例因子生成负载曲线。主要功能包括：
1. `LoadProfile` 类：用于创建和管理负载曲线。
    1. `__init__` 方法：初始化负载曲线生成器，设置步数、文件夹路径和负载名称。
    2. `create_file_with_daily` 方法：创建一个新的负载文件，并将其与日负载形状相关联。
    3. `add_redirect_and_mode_at_main_daily_dss` 方法：在主日负载文件中添加重定向和模式设置。
    4. `find_load_file_from` 方法：查找主负载文件中的负载文件。
    5. `find_load_names` 方法：查找主负载文件中的负载名称，并生成新的负载文件（如果需要）。
    6. `gen_loadprofile` 方法：生成负载曲线并保存到指定文件夹。
    7. `choose_loadprofile` 方法：选择特定的负载曲线文件。
    8. `get_loadprofile` 方法：获取特定负载曲线的数据。

Improve:
    1. 添加了 `interpolate_data` 方法，用于将原始负载曲线数据从原始点数插值到目标点数。
    2. 修改了 init 方法，添加了 `interpolate_to` 参数，用于指定插值后的步数。
    3. 修改了 `gen_loadprofile` 方法，适配了插值逻辑。

Note:
    1. 生成的负载曲线没有任何随机性
"""

import numpy as np
import pandas as pd
import os
from fnmatch import fnmatch

from scipy.interpolate import interp1d


class LoadProfile:
    """
        LoadProfile 类用于为系统中的负载生成负载曲线。
        它从指定的文件夹读取负载形状文件，并根据指定的步数和比例因子生成负载曲线。
    """

    def __init__(self, steps, dss_folder_path, dss_file, worker_idx=None,
                 interpolate_to=None,
                 interpolate_kind='linear'):
        self.steps = steps
        self.interpolate_to = interpolate_to  # 目标步数
        self.interpolate_kind = interpolate_kind  # 插值类型，缺省为'linear'

        self.dss_folder_path = dss_folder_path
        self.loadshape_path = os.path.join(dss_folder_path, 'loadshape')
        if worker_idx is None:
            self.loadshape_dss = 'loadshape.dss'
        else:
            self.loadshape_dss = 'loadshape_' + str(worker_idx) + '.dss'

        self.LOAD_NAMES = self.find_load_names(dss_file)

        self.FILES = []
        for f in os.listdir(self.loadshape_path):
            low = f.lower()
            if ('loadshape' in low) and low.endswith('.csv'):
                self.FILES.append(os.path.join(self.loadshape_path, f))

    def interpolate_data(self, data, original_steps:int, target_steps:int):
        """将原始负载曲线数据从原始点数插值到目标点数

        Args:
            data (array-like): 原始负载曲线数据
            original_steps (int): 原始数据的步数
            target_steps (int): 目标数据的步数
        Returns:
            array-like: 插值后的负载曲线数据
        """
        # 创建原始数据的时间点
        x_original = np.linspace(0, 1, original_steps)
        # 创建插值后的时间点
        x_target = np.linspace(0, 1, target_steps)
        # 创建插值函数
        interpolator = interp1d(x_original, data, kind=self.interpolate_kind)
        return interpolator(x_target)

    def create_file_with_daily(self, file_name):
        """
            创建一个新文件，名为为 'file_name[:-4]_daily.dss'
            在这个新文件中，如果有任何负载被创建，这个负载就会和它的日负载形状相关联。
        """
        fin = open(os.path.join(self.dss_folder_path, file_name), 'r', encoding='utf-8')
        fout = open(os.path.join(self.dss_folder_path, file_name[:-4] + '_daily.dss'), 'w', encoding='utf-8')
        for line in fin:
            if not line.lower().startswith('new load.') or ('daily' in line):
                fout.write(line)
            else:
                line = line.strip()
                if '!' in line: line = line[:line.find('!')].strip()  # remove inline comment
                if '//' in line: line = line[:line.find('//')].strip()  # remove inline comment
                spt = list(filter(None, line.split(' ')))  # filter out the empty string
                load = spt[1].split('.', 1)[1]
                fout.write(line + ' daily=loadshape_' + load + '\n')
        fin.close()
        fout.close()

    def add_redirect_and_mode_at_main_daily_dss(self, main_daily_dss):
        """
            Add redirect loadshape (& load file if any)
            and set daily mode at the main daily dss file

        Args:
            main_daily_dss: the file name of the main daily dss file

        Returns:
            the load dss file (if any) associated with the main dss file
        """
        # load the file
        fin = open(os.path.join(self.dss_folder_path, main_daily_dss), 'r', encoding='utf-8')
        lines = [line for line in fin]
        fin.close()

        # overwrite the file
        found_load, redirect_load = False, False
        load_file = None
        fout = open(os.path.join(self.dss_folder_path, main_daily_dss), 'w', encoding='utf-8')
        for line in lines:
            low = line.strip().lower()
            if '!' in low: low = low[:low.find('!')].strip()  # remove inline comment
            if '//' in low: low = low[:low.find('//')].strip()  # remove inline comment
            if (not found_load) and 'load' in low and not low.startswith('~'):
                fout.write('! add loadshape\n')
                fout.write('redirect ' + self.loadshape_dss + '\n\n')
                found_load = True

            low = low[:-4] if len(low) >= 4 else ''
            if (not redirect_load) and low.startswith('redirect'):
                if low.endswith('loads') or low.endswith('load'):
                    load_file = list(filter(None, line.strip().split(' ')))[1]  # remove the empty string
                    fout.write('redirect ' + load_file[:-4] + '_daily.dss\n')
                    redirect_load = True
                elif low.endswith('loads_daily') or low.endswith('load_daily'):
                    load_file = list(filter(None, line.strip().split(' ')))[1]  # remove the empty string
                    fout.write(line)
                    redirect_load = True
                else:
                    fout.write(line)
            else:
                fout.write(line)

        assert found_load, 'cannot find load at ' + main_daily_dss

        fout.write('Set mode=Daily number=1 hour=0 stepsize=3600 sec=0\n')
        fout.close()

        return load_file

    def find_load_file_from(self, main_dss):
        """从dss文件中查找负载文件
        Args:
            main_dss: 主dss文件名（字符串），用于查找其中重定向的负载文件
        Returns:
            load_file: 找到的负载文件名（字符串），如果未找到则为None
        """
        fin = open(os.path.join(self.dss_folder_path, main_dss), 'r', encoding='utf-8')
        load_file = None
        for line in fin:
            low = line.strip().lower()
            if '!' in low: low = low[:low.find('!')].strip()  # remove inline comment
            if '//' in low: low = low[:low.find('//')].strip()  # remove inline comment
            low = low[:-4] if len(low) >= 4 else ''
            # 找到第一个匹配的负载文件，退出
            if low.startswith('redirect') and (low.endswith('loads') or low.endswith('load')
                                               or low.endswith('loads_daily') or low.endswith(
                        'load_daily')):  # or 运算符的短路特性
                load_file = list(filter(None, line.strip().split(' ')))[1]
                break
        return load_file

    def find_load_names(self, main_dss):
        """
            Find the loads with daily loadshapes at main dss or the load dss files.
            If there is none,
                then generate new files (annotated _daily) with daily loadshapes.
        """

        def find_load_name(fname, names):
            file_path = os.path.join(self.dss_folder_path, fname)
            assert os.path.exists(file_path), file_path + ' not found'

            needs_load_daily, daily_mode = False, False
            with open(file_path, 'r', encoding='utf-8') as fin:
                for line in fin:
                    low = line.strip().lower()
                    if low.startswith('new load.'):
                        if 'daily' in low:
                            spt = line.split(' ')
                            spt = list(filter(None, spt))  # filter out the empty string
                            names.append(spt[1].split('.', 1)[1])
                        else:
                            needs_load_daily = True
                    if low.startswith('set mode=daily'):
                        daily_mode = True
            return needs_load_daily, daily_mode

        names = []

        # add from the main dss file
        needs_load_daily, daily_mode = find_load_name(main_dss, names)
        if needs_load_daily or (not daily_mode):
            ## Create a new _daily file. Add daily loadshape if needed
            self.create_file_with_daily(main_dss)

            ## add redirect and set daily mode at the new _daily file
            load_file = self.add_redirect_and_mode_at_main_daily_dss(main_dss[:-4] + '_daily.dss')
        else:
            load_file = self.find_load_file_from(main_dss)

        # add from the other load files
        if load_file is not None:
            if 'daily' in load_file:
                needs_load_daily, _ = find_load_name(load_file, names)
                assert (not needs_load_daily), 'invalid content in ' + load_file
            else:
                needs_load_daily, _ = find_load_name(load_file, names)
                if needs_load_daily: self.create_file_with_daily(load_file)

        # check empty or duplicate load
        assert len(
            names) > 0, 'daliy load not found. Consider modifying from the auto-generated file annotated with _daily'
        assert len(names) == len(set(names)), 'duplicate load names'

        return names

    def gen_loadprofile(self, scale=1.0):
        """
        为所有的负荷生成标幺化曲线并保存到指定文件夹. 重定义函数以匹配插值.
        Args:
            scale: 比例因子，用于缩放生成的负载曲线数据
        Returns:
            episodes: 生成的负载曲线的数量
        """
        # 读取负载曲线参考文件
        dfs = []
        for f in self.FILES:
            dfs.append(pd.read_csv(f, header=None))
        assert len(dfs) > 0, r'put load shapes files under ./loadshape'
        df = pd.concat(dfs).rename(columns={0: 'mul'}).reset_index(drop=True)
        if scale != 1.0:
            df['mul'] = df['mul'] * scale # df['mul'] 不是普通的 Python 列表，而是 Pandas 的 Series。对于 Series 或 Numpy 数组，* 运算符表示逐元素相乘（广播），不会扩充长度，只会让每个元素都乘以 scale。

        # 插值
        original_steps = self.steps
        if self.interpolate_to is not None:
            # 原始steps
            original_data = df['mul'].values
            interpolated_data = []

            # 分块插值
            for i in range(0, len(original_data), original_steps):
                chunk = original_data[i:i + original_steps]
                # 插值需要chunk长度等于原始steps
                if len(chunk) == original_steps:
                    interpolated_chunk = self.interpolate_data(chunk, original_steps, self.interpolate_to)
                    interpolated_data.extend(interpolated_chunk)

            # 更新df和steps
            df = pd.DataFrame({'mul': interpolated_data})
            self.steps = self.interpolate_to

        # 计算生成的数据可分隔为多少个episodes
        episodes = len(df) // (self.steps * len(self.LOAD_NAMES))

        # 检查loadprofile文件夹是否存在
        checks = [fnmatch(f, '0*') for f in os.listdir(self.loadshape_path)]
        scale_txt = os.path.join(self.loadshape_path, 'scale.txt')
        fscale = np.genfromtxt(scale_txt).reshape(1)[0] if os.path.exists(scale_txt) else None
        if sum(checks) == episodes and fscale == scale:
            return episodes

        # 保存当前的scale
        np.savetxt(scale_txt, np.array([scale]))

        # 插入loadname, day, step列
        load_col, episode_col, step_col = [], [], []
        for i in range(self.steps * episodes * len(self.LOAD_NAMES)):
            load_col.append(self.LOAD_NAMES[i // (self.steps * episodes)])
            episode_col.append((i // self.steps) % episodes)
            step_col.append(i % self.steps)
        df = df[:len(load_col)]
        df['load'] = load_col
        df['episode'] = episode_col
        df['step'] = step_col

        # 排序并输出
        df = df.sort_values(by=['episode', 'load', 'step'])[['episode', 'load', 'step', 'mul']].reset_index(drop=True)
        for episode in range(episodes):
            if not os.path.exists(os.path.join(self.loadshape_path, str(episode).zfill(4))):
                os.mkdir(os.path.join(self.loadshape_path, str(episode).zfill(4)))
            sdf = df[df['episode'] == episode]
            for load in self.LOAD_NAMES:
                series = sdf[sdf['load'] == load]['mul']
                series.to_csv(
                    os.path.join(self.loadshape_path, str(episode).zfill(4), load + '.csv'),
                    header=False, index=False)

        return episodes

    def choose_loadprofile(self, idx):
        """
        选择特定的负载曲线文件，并在主负载文件中添加重定向和模式设置.(修改了zfill参数3->4)
        Args:
            idx (int): 负载曲线编号
        Returns:
            负载曲线文件的路径
        """
        assert os.path.exists(os.path.join(self.loadshape_path, str(idx).zfill(4))), 'idx does not exist'

        with open(os.path.join(self.dss_folder_path, self.loadshape_dss), 'w') as fp:
            for load_name in self.LOAD_NAMES:
                fp.write(
                    f'New Loadshape.loadshape_{load_name} npts={self.steps} sinterval={60 * 60 * 24 // self.steps} ' +
                    'mult=(file=./' + os.path.join('loadshape', str(idx).zfill(4), load_name + '.csv') + ')\n')

        return os.path.join(self.loadshape_path, str(idx).zfill(4))

    def get_loadprofile(self, idx):
        """
        获取指定编号目录的所有负载曲线数据.(修改了zfill参数3->4)
        Args:
            idx: loadrrofile的编号，整数
        Returns:
            all_loads: 包含所有(修饰负载而非时间)负载的曲线数据的DataFrame
        """
        folder_path = os.path.join(self.loadshape_path, str(idx).zfill(4))
        csv_paths = os.listdir(folder_path)
        temp_loads = []
        # 遍历编号目录下的所有csv文件
        for csv in csv_paths:
            csv_file = os.path.join(folder_path, csv)
            load = pd.read_csv(csv_file, header=None, names=[csv.split('.')[0]])
            temp_loads.append(load)

        all_loads = pd.concat(temp_loads, axis=1)
        return all_loads
