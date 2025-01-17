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

        # 如果是 NaN，则存为 None，以便后续识别为无效(陆地)
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
    计算某个邻居点 (nx, ny) 向当前点移动的概率贡献：
    - 若邻居不存在或为陆地，返回 0
    - 贡献 = 邻居的概率 * 邻居在 direction_key 方向的移动概率
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
    基于 current_probs 计算下一时刻的概率分布。返回新的字典 next_probs。
    """
    next_probs = {}
    for (x, y), cell_info in grid_map.items():
        if is_land(cell_info):
            # 陆地，概率直接为 0
            next_probs[(x, y)] = 0.0
            continue

        total = 0.0
        # (x, y+1) 向下 -> (x, y)
        total += get_neighbor_contribution(x, y + 1, 'P_down', current_probs, grid_map)
        # (x, y-1) 向上 -> (x, y)
        total += get_neighbor_contribution(x, y - 1, 'P_up', current_probs, grid_map)
        # (x-1, y) 向右 -> (x, y)
        total += get_neighbor_contribution(x - 1, y, 'P_right', current_probs, grid_map)
        # (x+1, y) 向左 -> (x, y)
        total += get_neighbor_contribution(x + 1, y, 'P_left', current_probs, grid_map)

        # 简单处理一下浮点精度
        total = float(f"{total:.8f}")
        next_probs[(x, y)] = total
    return next_probs


def print_probability_distribution(probabilities, start_x, start_y, range_val=5):
    """
    将 (start_x, start_y) 附近 ±range_val 的概率打印出来，方便在命令行查看。
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


def plot_heatmap(probabilities, grid_map, start_x, start_y, time_idx=0, range_val=5):
    """
    以热力图的形式可视化在 (start_x, start_y) 附近±range_val 的概率分布；
    time_idx 表示当前时刻，用于标题显示。
    """
    min_x = start_x - range_val
    max_x = start_x + range_val
    min_y = start_y - range_val
    max_y = start_y + range_val

    width = max_x - min_x + 1
    height = max_y - min_y + 1

    # 创建一个 2D 数组来保存绘图数据；默认 0
    data = np.zeros((height, width), dtype=float)

    # 同时记录哪些位置是陆地
    land_mask = np.zeros((height, width), dtype=bool)

    # 填充 data
    # 注意：我们希望行 0 对应 max_y，行 height-1 对应 min_y
    for row_idx, y in enumerate(range(max_y, min_y - 1, -1)):
        for col_idx, x in enumerate(range(min_x, max_x + 1)):
            cell_info = grid_map.get((x, y))
            prob = probabilities.get((x, y), 0.0)

            if (cell_info is None) or is_land(cell_info):
                # 标记为陆地
                land_mask[row_idx, col_idx] = True
            else:
                data[row_idx, col_idx] = prob

    plt.figure(figsize=(6, 6))
    plt.title(f"Probability Heatmap (t={time_idx})")

    # cmap 设置为 'Blues'：概率越大颜色越深
    img = plt.imshow(data, cmap='Blues', interpolation='nearest')
    plt.colorbar(img, label='Probability')

    # 将陆地格子覆盖为灰色
    for row_idx in range(height):
        for col_idx in range(width):
            if land_mask[row_idx, col_idx]:
                plt.gca().add_patch(
                    plt.Rectangle((col_idx - 0.5, row_idx - 0.5), 1, 1,
                                  fill=True, color='gray', alpha=1)
                )

    # 坐标轴标签
    plt.xticks(
        ticks=np.arange(0, width),
        labels=[str(x) for x in range(min_x, max_x + 1)],
        rotation=90
    )
    plt.yticks(
        ticks=np.arange(0, height),
        labels=[str(y) for y in range(max_y, min_y - 1, -1)]
    )

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()
    plt.show()
    plt.close()  # 清除当前图，防止与下一时刻叠加


def main():
    csv_path = "Grid_with_Integer_Coordinates.csv"
    grid_map = load_data(csv_path)

    # 设置迭代次数 t
    time_steps = 5

    # 初始位置
    start_x, start_y = 43, 4

    # 初始化 t=0 的概率分布
    current_probs = init_probabilities(grid_map, start_x, start_y)

    for t in range(time_steps):
        print(f"=== t={t} 时刻的概率 (周围5格) ===")
        print_probability_distribution(current_probs, start_x, start_y, range_val=5)

        # 画出当前时刻的热力图
        plot_heatmap(current_probs, grid_map, start_x, start_y, time_idx=t, range_val=5)

        # 计算下一时刻概率
        # 当循环到最后一次时，就不必再计算下一时刻了（除非你要再看 t=time_steps 时刻图）
        if t < time_steps - 1:
            next_probs = calculate_next_probabilities(current_probs, grid_map)
            current_probs = next_probs


if __name__ == "__main__":
    main()