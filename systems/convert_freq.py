# Copyright 2025 Su Chenhui
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-DSSGym-Commercial

"""
将线路参数的频率从60Hz转换为50Hz
"""

import re
import sys


def convert_linecode_60Hz_to_50Hz(input_file, output_file):
    """
    将60Hz的线路参数(特别是电抗矩阵)转换为50Hz

    参数:
    input_file (str): 输入文件路径，包含60Hz的线路代码定义
    output_file (str): 输出文件路径，将包含转换为50Hz的线路代码定义
    """
    # 定义正则表达式，用于识别BaseFreq, xmatrix和rmatrix行
    basefreq_pattern = re.compile(r'(BaseFreq\s*=\s*)(\d+)', re.IGNORECASE)
    xmatrix_pattern = re.compile(r'~\s*xmatrix\s*=\s*(\[.*?\]|\(.*?\))', re.IGNORECASE | re.DOTALL)  # Claude3.7的狡辩：尽管在r-string中 \[ 和 \] 技术上是冗余的转义，但它们在这里是完全可以接受的，并不会影响表达式的功能。

    # 读取输入文件
    with open(input_file, 'r') as f:
        content = f.read()

    # 替换BaseFreq从60Hz到50Hz
    content = basefreq_pattern.sub(r'\g<1>50', content)

    # 查找并处理所有xmatrix定义
    def process_xmatrix(match):
        xmatrix_str = match.group(1)
        # 检查是否是方括号或圆括号格式
        is_bracket = xmatrix_str.startswith('[')

        # 移除括号以处理内部内容
        xmatrix_inner = xmatrix_str[1:-1].strip()

        # 处理矩阵元素
        elements = []
        # 处理矩阵中的每个部分（由|分隔）
        parts = xmatrix_inner.split('|')
        for part in parts:
            # 处理每个部分中的数字
            nums = re.findall(r'[-+]?\d*\.\d+|\d+', part)
            # 将每个数乘以50/60因子（电抗与频率成正比）
            scaled_nums = [str(float(num) * (50 / 60)) for num in nums]
            # 保持原有的空格格式
            # 寻找数字在原始字符串中的位置并替换
            new_part = part
            for i, num in enumerate(nums):
                num_pos = new_part.find(num)
                if num_pos != -1:
                    new_part = new_part[:num_pos] + scaled_nums[i] + new_part[num_pos + len(num):]
            elements.append(new_part)

        # 重新组合矩阵字符串
        new_xmatrix_inner = ' | '.join(elements)

        # 重新添加括号
        open_bracket = '[' if is_bracket else '('
        close_bracket = ']' if is_bracket else ')'
        new_xmatrix_str = open_bracket + new_xmatrix_inner + close_bracket

        return '~ xmatrix = ' + new_xmatrix_str

    # 应用处理函数到所有xmatrix定义
    content = xmatrix_pattern.sub(process_xmatrix, content)

    # 将处理后的内容写入输出文件
    with open(output_file, 'w') as f:
        f.write(content)

    print(f"转换完成！60Hz线路参数已转换为50Hz，并保存到 {output_file}")


# 如果作为独立脚本运行
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法: python convert_linecodes.py <输入文件> <输出文件>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    convert_linecode_60Hz_to_50Hz(input_file, output_file)
