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
        # 上方 (x, y+1) 向下 -> (x, y)
        total += get_neighbor_contribution(x, y + 1, 'P_down', current_probs, grid_map)
        # 下方 (x, y-1) 向上 -> (x, y)
        total += get_neighbor_contribution(x, y - 1, 'P_up', current_probs, grid_map)
        # 左方 (x-1, y) 向右 -> (x, y)
        total += get_neighbor_contribution(x - 1, y, 'P_right', current_probs, grid_map)
        # 右方 (x+1, y) 向左 -> (x, y)
        total += get_neighbor_contribution(x + 1, y, 'P_left', current_probs, grid_map)

        # 简单处理一下浮点精度
        total = float(f"{total:.8f}")
        next_probs[(x, y)] = total
    return next_probs


def build_data_array(probabilities, grid_map, start_x, start_y, range_val=5):
    """
    构造 2D Numpy 数组 data (height x width)，
    其中 data[row, col] = 对应坐标的概率值，如果是陆地则为 0。
    同时返回一个 land_mask 记录哪些格是陆地 (True)。
    """
    min_x = start_x - range_val
    max_x = start_x + range_val
    min_y = start_y - range_val
    max_y = start_y + range_val

    width = max_x - min_x + 1
    height = max_y - min_y + 1

    data = np.zeros((height, width), dtype=float)
    land_mask = np.zeros((height, width), dtype=bool)

    # 行 0 对应 max_y，行 height-1 对应 min_y
    for row_idx, y in enumerate(range(max_y, min_y - 1, -1)):
        for col_idx, x in enumerate(range(min_x, max_x + 1)):
            cell_info = grid_map.get((x, y))
            prob = probabilities.get((x, y), 0.0)
            if (cell_info is None) or is_land(cell_info):
                land_mask[row_idx, col_idx] = True
            else:
                data[row_idx, col_idx] = prob

    return data, land_mask


def main():
    csv_path = "Grid_with_Integer_Coordinates.csv"
    grid_map = load_data(csv_path)

    # 自动演示多少个时刻
    time_steps = 5

    # 初始位置
    start_x, start_y = 43, 4
    range_val = 5

    # 初始化 t=0 的概率分布
    current_probs = init_probabilities(grid_map, start_x, start_y)

    # -- 首次绘图设置 (只创建一次 Figure 和 Axes) --
    plt.ion()  # 打开交互模式
    fig, ax = plt.subplots(figsize=(6, 6))

    # 构建初始 data 数组和 land_mask
    data, land_mask = build_data_array(current_probs, grid_map, start_x, start_y, range_val=range_val)

    # 显示初始热力图
    img = ax.imshow(data, cmap='Blues', interpolation='nearest')
    cbar = plt.colorbar(img, ax=ax, label='Probability')

    # 叠加陆地的灰色方块(只做一次，保持不变)
    height, width = data.shape
    for row_idx in range(height):
        for col_idx in range(width):
            if land_mask[row_idx, col_idx]:
                # 用灰色方块覆盖陆地
                ax.add_patch(
                    plt.Rectangle((col_idx - 0.5, row_idx - 0.5), 1, 1,
                                  fill=True, color='gray', alpha=1)
                )

    # 设置坐标轴(只做一次)
    min_x = start_x - range_val
    max_x = start_x + range_val
    min_y = start_y - range_val
    max_y = start_y + range_val

    # X 方向
    ax.set_xticks(np.arange(0, width))
    ax.set_xticklabels([str(x) for x in range(min_x, max_x + 1)], rotation=90)
    # Y 方向
    ax.set_yticks(np.arange(0, height))
    ax.set_yticklabels([str(y) for y in range(max_y, min_y - 1, -1)])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    plt.tight_layout()
    # 在同一窗口里动态刷新
    plt.show(block=False)

    # -- 循环迭代，每次 0.5s 更新一次同一个图像 --
    for t in range(time_steps):
        ax.set_title(f"Probability Heatmap (t={t})")
        print(f"=== t={t} ===")
        print_probability_distribution(current_probs, start_x, start_y, range_val)

        # 若不是第一次( t>0 )，需要更新 data
        if t > 0:
            # 已经在上一循环里更新了 current_probs，这里只需把新的数据放进图里
            data, _ = build_data_array(current_probs, grid_map, start_x, start_y, range_val)
            img.set_data(data)  # 更新热力图数据

        plt.draw()
        plt.pause(0.5)  # 暂停 0.5s

        # 计算下一时刻
        if t < time_steps - 1:
            next_probs = calculate_next_probabilities(current_probs, grid_map)
            current_probs = next_probs

    # 最后可以让图保持打开，等待手动关闭
    # 或者程序到此结束后，如果你执行脚本就会退出。
    input("Press Enter to exit...")


def print_probability_distribution(probabilities, start_x, start_y, range_val=5):
    """
    在命令行打印 (start_x, start_y) 附近 ±range_val 的概率值。
    """
    min_x = start_x - range_val
    max_x = start_x + range_val
    min_y = start_y - range_val
    max_y = start_y + range_val

    for y in range(max_y, min_y - 1, -1):
        row_str = []
        for x in range(min_x, max_x + 1):
            p = probabilities.get((x, y), 0.0)
            row_str.append(f"{p:.4f}")
        print(" ".join(row_str))
    print()


if __name__ == "__main__":
    main()
