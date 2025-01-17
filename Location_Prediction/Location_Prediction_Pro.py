import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_data(csv_path):
    """
    读取 CSV 文件，并将每个点的信息存储到字典 grid_map 中：
    key: (x, y)
    value: {'P_up': float 或 None, 'P_down': ..., 'P_left': ..., 'P_right': ...}
    其中 None 表示该方向无效(陆地)。
    """
    df = pd.read_csv(csv_path)
    grid_map = {}
    for _, row in df.iterrows():
        x = int(row['x'])
        y = int(row['y'])

        def to_none_if_nan(val):
            return None if pd.isna(val) else val

        P_up = to_none_if_nan(row['P_up'])
        P_down = to_none_if_nan(row['P_down'])
        P_left = to_none_if_nan(row['P_left'])
        P_right = to_none_if_nan(row['P_right'])

        grid_map[(x, y)] = {
            'P_up': P_up,
            'P_down': P_down,
            'P_left': P_left,
            'P_right': P_right
        }
    return grid_map

def is_land(cell_info):
    """
    如果四个方向概率均为 None，则视为陆地。
    """
    return (
        cell_info['P_up'] is None and
        cell_info['P_down'] is None and
        cell_info['P_left'] is None and
        cell_info['P_right'] is None
    )

def init_probabilities(grid_map, start_x, start_y):
    """
    初始化概率分布：
    - (start_x, start_y) = 1
    - 其他点 = 0
    返回一个字典 probabilities: {(x, y): float, ...}
    """
    probabilities = {}
    for (x, y) in grid_map.keys():
        if x == start_x and y == start_y:
            probabilities[(x, y)] = 1.0
        else:
            probabilities[(x, y)] = 0.0
    return probabilities

def get_neighbor_contribution(nx, ny, direction_key, current_probs, grid_map):
    """
    计算某个邻居点 (nx, ny) 向当前点移动的概率贡献。
    """
    neighbor_info = grid_map.get((nx, ny))
    if neighbor_info is None:
        return 0.0
    dir_val = neighbor_info[direction_key]
    if dir_val is None:
        return 0.0
    neighbor_prob = current_probs.get((nx, ny), 0.0)
    return neighbor_prob * dir_val

def calculate_next_probabilities(current_probs, grid_map):
    """
    基于 current_probs 计算下一时刻的概率分布。
    返回新的字典 next_probs。
    """
    next_probs = {}
    for (x, y), cell_info in grid_map.items():
        if is_land(cell_info):
            next_probs[(x, y)] = 0.0
            continue

        total = 0.0
        # 上方 (x, y+1) -> (x, y)
        total += get_neighbor_contribution(x, y+1, 'P_down', current_probs, grid_map)
        # 下方 (x, y-1) -> (x, y)
        total += get_neighbor_contribution(x, y-1, 'P_up', current_probs, grid_map)
        # 左方 (x-1, y) -> (x, y)
        total += get_neighbor_contribution(x-1, y, 'P_right', current_probs, grid_map)
        # 右方 (x+1, y) -> (x, y)
        total += get_neighbor_contribution(x+1, y, 'P_left', current_probs, grid_map)

        total = float(f"{total:.8f}")
        next_probs[(x, y)] = total
    return next_probs

def build_data_array(probabilities, grid_map, center_x, center_y, range_val=5):
    """
    以 (center_x, center_y) 为中心，构造 2D Numpy 数组 data (height x width)，
    同时返回 land_mask 用于标识陆地，及 (min_x, max_x, min_y, max_y)。
    """
    min_x = center_x - range_val
    max_x = center_x + range_val
    min_y = center_y - range_val
    max_y = center_y + range_val

    width = max_x - min_x + 1
    height = max_y - min_y + 1

    data = np.zeros((height, width), dtype=float)
    land_mask = np.zeros((height, width), dtype=bool)

    for row_idx, y in enumerate(range(max_y, min_y - 1, -1)):
        for col_idx, x in enumerate(range(min_x, max_x + 1)):
            cell_info = grid_map.get((x, y))
            prob = probabilities.get((x, y), 0.0)
            if (cell_info is None) or is_land(cell_info):
                land_mask[row_idx, col_idx] = True
            else:
                data[row_idx, col_idx] = prob

    return data, land_mask, (min_x, max_x, min_y, max_y)

def main():
    csv_path = "Grid_with_Integer_Coordinates.csv"
    grid_map = load_data(csv_path)

    # 迭代次数
    time_steps = 100
    start_x, start_y = 43, 4
    current_probs = init_probabilities(grid_map, start_x, start_y)

    plt.ion()
    fig = plt.figure(figsize=(7, 7))

    for t in range(time_steps):
        # 找到概率最大的坐标
        max_cell, max_val = max(current_probs.items(), key=lambda x: x[1])
        center_x, center_y = max_cell

        # 构建当前显示区域
        data, land_mask, (min_x, max_x, min_y, max_y) = build_data_array(
            current_probs, grid_map, center_x, center_y, range_val=5
        )

        # 动态设置 vmax
        cur_max = np.max(data)
        if cur_max <= 0:
            cur_max = 1e-6

        # 彻底清空 figure
        fig.clf()

        # 重新创建 Axes
        ax = fig.add_subplot(111)

        # 设置标题并使用 Times New Roman
        ax.set_title(
            f"t={t*15} min MaxProb={max_val:.6f} at ({center_x},{center_y})",
            fontname="Microsoft YaHei",    # <-- 这里！
            fontsize=14                    # 可根据需要调大小
        )

        # 绘制热力图
        img = ax.imshow(
            data,
            cmap='Blues',
            interpolation='nearest',
            vmin=0,
            vmax=cur_max
        )
        # 创建 Colorbar，并设置 label 字体为 Times New Roman
        cbar = fig.colorbar(img, ax=ax)
        cbar.set_label("", fontname="Times New Roman", fontsize=12)

        # 覆盖陆地为灰色
        height, width = data.shape
        for row_idx in range(height):
            for col_idx in range(width):
                if land_mask[row_idx, col_idx]:
                    ax.add_patch(
                        plt.Rectangle((col_idx - 0.5, row_idx - 0.5), 1, 1,
                                      fill=True, color='gray', alpha=1)
                    )

        # 标注 (x, y) 坐标
        for row_idx, yv in enumerate(range(max_y, min_y - 1, -1)):
            for col_idx, xv in enumerate(range(min_x, max_x + 1)):
                ax.text(
                    col_idx,
                    row_idx,
                    f"({xv},{yv})",
                    ha='center',
                    va='center',
                    fontsize=7,
                    color='black'
                )

        # 设置坐标轴
        ax.set_xticks(np.arange(0, width))
        ax.set_xticklabels([str(xx) for xx in range(min_x, max_x + 1)])
        ax.set_yticks(np.arange(0, height))
        ax.set_yticklabels([str(yy) for yy in range(max_y, min_y - 1, -1)])
        ax.set_xlabel("")
        ax.set_ylabel("")

        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.5)

        # 计算下一时刻
        if t < time_steps - 1:
            next_probs = calculate_next_probabilities(current_probs, grid_map)
            current_probs = next_probs

    input("Press Enter to exit...")

if __name__ == "__main__":
    main()
