import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_csv("Grid_with_Integer_Coordinates.csv")

# 重塑数据为一个 63x152 的网格（行: 0-62, 列: 0-151）
grid = np.zeros((63, 152, 2))  # 使用三维数组来存储方向性 (x, y)

# 填充网格数据，计算方向向量
for index, row in data.iterrows():
    x, y = int(row['x']), int(row['y'])  # 确保 x 和 y 是整数

    # 方向向量：上、下、左、右
    up = (0, 1)
    down = (0, -1)
    left = (-1, 0)
    right = (1, 0)

    # 计算方向向量的合成
    v_x = row['P_right'] - row['P_left']  # 右方向 - 左方向
    v_y = row['P_up'] - row['P_down']  # 上方向 - 下方向

    # 存储合成的方向向量
    grid[x, y] = np.array([v_x, v_y])

# 计算方向向量的大小（模长）
magnitude = np.linalg.norm(grid, axis=2)

# 标准化方向向量，确保箭头长度为单位长度
v_x_normalized = grid[:, :, 0] / magnitude  # 东向速度（x方向）
v_y_normalized = grid[:, :, 1] / magnitude  # 北向速度（y方向）


# 创建可视化函数
def visualize_grid(grid, v_x_normalized, v_y_normalized):
    plt.figure(figsize=(12, 8))

    # 使用 quiver 绘制箭头
    plt.quiver(np.arange(grid.shape[1]), np.arange(grid.shape[0]), v_x_normalized, v_y_normalized, angles='xy',
               scale_units='xy', scale=1, color='blue')

    # 设置图形的标题和标签
    plt.title("Directionality with Unit Vectors", fontsize=16)
    plt.xlabel("Longitude", fontsize=12)
    plt.ylabel("Latitude", fontsize=12)

    # 设置坐标轴范围
    plt.xlim(0, grid.shape[1] - 1)
    plt.ylim(0, grid.shape[0] - 1)

    # 反转 y 轴，使其与网格坐标系一致
    plt.gca().invert_yaxis()

    # 显示图形
    plt.show()


# 调用可视化函数
visualize_grid(grid, v_x_normalized, v_y_normalized)
