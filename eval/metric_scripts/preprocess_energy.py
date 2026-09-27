import os
from collections import defaultdict
import argparse

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl

plt.switch_backend('agg')
plt.style.use('seaborn-paper')

# mpl.font_manager.fontManager.addfont(
#         os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Helvetica-Regular.ttf'))

# plt.rcParams['font.family'] = 'Helvetica Neue LT'
plt.rcParams['legend.fontsize'] = 12.0
plt.rcParams['axes.labelsize'] = 14.0
plt.rcParams['xtick.labelsize'] = 14.0
plt.rcParams['ytick.labelsize'] = 14.0
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['savefig.dpi'] = 200.0

color_pool = ['blue', 'red', 'darkorange', 'purple', 'black', 'cyan', 'lime', 'gold']
def remove_outliers(data):
    """
    Identifies and removes outliers from the data set using the IQR method.
    """
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return [x for x in data if lower_bound <= x <= upper_bound]

def parse_log(path, fields):
    data = defaultdict(list)
    def _parse_line(line):
        flags = line.split(':')
        if len(flags) >= 2 and any(map(lambda x: flags[-2].endswith(x), fields)):
            k = flags[-2].split('/')[1]
            v = float(flags[-1].strip())
            data[k].append(v)
            
    with open(path) as f:
        for line in f:
            _parse_line(line)

    return data


def parse_energy_log(energy_log_path):
    fields = ['ddG']
    data = defaultdict(list)
    key = []
    def _parse_line(line):
        flags = line.split(':')
        if len(flags) >= 2:
            # print(f"flags: {flags}")
            
            k = (flags[-2].split(' ')[-1]).split('.')[0]
            k = k.split('@')[0]
            v = float(flags[-1].strip())

            data[k] = v
            # print(f"k: {k}; v: {v}")
            key.append(k)
    with open(energy_log_path) as f:
        for line in f:
            if 'dG_wild' in line:
                _parse_line(line)
    return data, key


def get_data(log_dir, rank=None):
    data = []
    
    energy_log_path = '/home/zhutian/data/antibody/native_energy.log'
    # Parse the metric.log file and retrieve relevant information
    energy, key = parse_energy_log(energy_log_path)
    # Add the parsed metric to the list of all metrics
    list_of_tuples = list(energy.items())
    data = pd.DataFrame(list_of_tuples, columns=['Name', 'Energy'])
    Q1 = data['Energy'].quantile(0.25)
    Q3 = data['Energy'].quantile(0.75)
    IQR = Q3 - Q1

    # 定义边界
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # 过滤异常值
    filtered_data = data[(data['Energy'] >= lower_bound) & (data['Energy'] <= upper_bound)]
    unique_values_in_column_A = filtered_data['Name'].unique()
    print(f"unique values: {len(unique_values_in_column_A)}")
    # print(f"new_data: {filtered_data.unique().describe()}")
    filtered_data.to_csv('./Energy.csv', index=False)


    return filtered_data


def main(args):
    figure_path = os.path.join(args.output_dir, 'energy.pdf')

    energy_data = get_data(args.log_dir)
    # plot_curve(energy_data,figure_path )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    
    main(args)


# 这段代码的目的是从一个包含蛋白质能量评估结果的日志文件中读取 dG_wild 能量项，剔除异常值后输出为 CSV 文件，用于后续分析，比如结构能量分布可视化或抗体设计优化基准。

# 🧠 总体含义总结（物理与数学视角）
# 物理背景：dG_wild 是 wild-type 抗体结构与抗原复合物的界面自由能（ΔG），衡量结合稳定性。越负越稳定，越正越不稳定。

# 数学方法：使用 IQR（四分位距）方法 去除能量分布中的异常值，保留中间 50% ± 1.5 倍 IQR 范围的数据。

# 程序功能：读取日志 → 提取能量数据 → 去除异常值 → 保存为 CSV 文件，供后续统计绘图等使用。