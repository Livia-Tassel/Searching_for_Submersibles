import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# 从文件加载最后的概率矩阵
prob_matrix = pd.read_csv('recursive_prob_matrix.csv', header=None).values

# 归一化矩阵数据，将其映射到[0, 1]范围
norm = Normalize(vmin=np.min(prob_matrix), vmax=np.max(prob_matrix))

# 创建一个热力图
plt.figure(figsize=(8, 6))

# 使用imshow来展示热力图，加入归一化处理
plt.imshow(prob_matrix, cmap='viridis', origin='lower', interpolation='nearest', norm=norm)

# 添加颜色条
plt.colorbar(label='Probability')

# 添加标题和标签
plt.title('2D Probability Distribution')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')

# 可选：放宽概率阈值以增加图像的多样性（例如，允许一些小概率项更好地可视化）
# 例如，降低归一化的范围，使得低概率部分也能更明显地展示
threshold = 0.1  # 设置阈值
prob_matrix[prob_matrix < threshold] = 0  # 把低于阈值的概率设为0

# 重新进行归一化
norm = Normalize(vmin=np.min(prob_matrix), vmax=np.max(prob_matrix))

# 重新绘制热力图
plt.imshow(prob_matrix, cmap='viridis', origin='lower', interpolation='nearest', norm=norm)

# 再次添加颜色条
plt.colorbar(label='Probability')


# 展示图形
plt.show()
