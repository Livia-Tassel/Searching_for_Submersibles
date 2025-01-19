import numpy as np
import pandas as pd

# 从CSV文件读取数据
data = pd.read_csv('Grid_with_Integer_Coordinates.csv')

# 这里假设：最大 x -> rows，最大 y -> cols
rows = data['x'].max() + 1
cols = data['y'].max() + 1

# 初始化四个方向的概率矩阵，全填 NaN
P_up = np.full((rows, cols), np.nan)
P_down = np.full((rows, cols), np.nan)
P_left = np.full((rows, cols), np.nan)
P_right = np.full((rows, cols), np.nan)

# 根据 CSV 表格往四个方向的概率矩阵里填值
for idx, row in data.iterrows():
    x, y = int(row['x']), int(row['y'])
    P_up[x, y] = row['P_up']
    P_down[x, y] = row['P_down']
    P_left[x, y] = row['P_left']
    P_right[x, y] = row['P_right']

# 初始化潜航器的起始概率矩阵
prob_matrix = np.zeros((rows, cols))

# 如果起始坐标是(43, 4)，且 x=43 不会越界：
# 注意：prob_matrix[x, y] = prob_matrix[43, 4]
# 以你的CSV数据大小为准，确认 (43,4) 的确可用
prob_matrix[3, 4] = 1.0

# 定义一个函数，用于进行一次递归传播
def propagate_probabilities(current_prob_matrix):
    # 新矩阵先拷贝一份，以保证对陆地保持 np.nan
    new_prob_matrix = np.full((rows, cols), np.nan)

    # 遍历所有水域坐标 (即非 NaN)
    for x in range(rows):
        for y in range(cols):
            if np.isnan(P_up[x, y]):
                # 陆地或者无效点，保持 NaN 不变
                continue
            else:
                # 该点是水域，初始化为0，准备接收周围贡献
                new_prob_matrix[x, y] = 0.0

                # 1) 来自上方格子 (x, y-1) 的贡献
                if y - 1 >= 0 and not np.isnan(P_up[x, y - 1]):
                    # 如果上方格子也是水域，表示可以向下移动到 (x, y)
                    # sub 在 (x, y-1) 时，想要向下 -> P_down[x, y-1]
                    new_prob_matrix[x, y] += current_prob_matrix[x, y - 1] * P_down[x, y - 1]

                # 2) 来自下方格子 (x, y+1) 的贡献
                if y + 1 < cols and not np.isnan(P_up[x, y + 1]):
                    # 下方格子是水域，可以向上移动到 (x, y)
                    # sub 在 (x, y+1) 时，想要向上 -> P_up[x, y+1]
                    new_prob_matrix[x, y] += current_prob_matrix[x, y + 1] * P_up[x, y + 1]

                # 3) 来自左方格子 (x-1, y) 的贡献
                if x - 1 >= 0 and not np.isnan(P_up[x - 1, y]):
                    # 左方格子是水域，可以向右移动到 (x, y)
                    # sub 在 (x-1, y) 时，想要向右 -> P_right[x-1, y]
                    new_prob_matrix[x, y] += current_prob_matrix[x - 1, y] * P_right[x - 1, y]

                # 4) 来自右方格子 (x+1, y) 的贡献
                if x + 1 < rows and not np.isnan(P_up[x + 1, y]):
                    # 右方格子是水域，可以向左移动到 (x, y)
                    # sub 在 (x+1, y) 时，想要向左 -> P_left[x+1, y]
                    new_prob_matrix[x, y] += current_prob_matrix[x + 1, y] * P_left[x + 1, y]

    return new_prob_matrix

# 递归次数
t = 100

current_prob_matrix = prob_matrix.copy()
for _ in range(t):
    current_prob_matrix = propagate_probabilities(current_prob_matrix)

# 将最终的概率矩阵保存到CSV文件
output_df = pd.DataFrame(current_prob_matrix)
# float_format 设置想要的精度与格式，na_rep 表示将 NaN 输出为什么
output_df.to_csv('recursive_prob_matrix.csv',
                 index=False, header=False, float_format='%.8f', na_rep='NaN')

print("递归预测的最终概率矩阵已保存为 'recursive_prob_matrix.csv'.")
