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

# 提取区域 x[5, 35], y[5, 35]
x_min, x_max = 5, 35
y_min, y_max = 5, 35

# 截取区域
v_x_region = v_x_normalized[y_min:y_max + 1, x_min:x_max + 1]
v_y_region = v_y_normalized[y_min:y_max + 1, x_min:x_max + 1]


# 创建可视化函数
def visualize_grid_region(v_x_region, v_y_region):
    fig, ax = plt.subplots(figsize=(8, 8))

    # 统一使用 'fancy' 箭头样式
    for i in range(v_x_region.shape[0]):
        for j in range(v_x_region.shape[1]):
            # 计算箭头的起点位置和方向
            x_pos = j + 0.5
            y_pos = i + 0.5
            dx = v_x_region[i, j]
            dy = v_y_region[i, j]

            # 绘制箭头，使用 'fancy' 样式
            ax.annotate('', xy=(x_pos + dx, y_pos + dy), xytext=(x_pos, y_pos),
                        arrowprops=dict(arrowstyle='fancy', color='#81B2DF', lw=1))

    # 设置图形的标题和标签，优化字体
    ax.set_title("Directionality with Unit Vectors (Region)", fontsize=18, family='Times New Roman', fontweight='bold')
    ax.set_xlabel("Longitude (Subset)", fontsize=14, family='Times New Roman', fontweight='bold')
    ax.set_ylabel("Latitude (Subset)", fontsize=14, family='Times New Roman', fontweight='bold')

    # 设置坐标轴范围
    ax.set_xlim(0, v_x_region.shape[1])
    ax.set_ylim(0, v_y_region.shape[0])

    # 反转 y 轴，使其与网格坐标系一致
    ax.invert_yaxis()

    # 隐藏坐标轴
    ax.axis('off')

    # 显示图形
    plt.show()


# 调用可视化函数
visualize_grid_region(v_x_region, v_y_region)
