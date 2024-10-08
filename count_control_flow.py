import matplotlib.pyplot as plt
import numpy as np

# 生成随机数据
codebase_sizes = np.random.randint(1000, 5000, 10)  # 生成 10 个代码库规模在 1000 到 5000 之间的随机数
control_flow_counts = np.random.randint(50, 200, 10)  # 生成 10 个控制流数量在 50 到 200 之间的随机数

# 绘制柱状图
plt.figure(figsize=(10, 6))
plt.bar(codebase_sizes, control_flow_counts, color='skyblue')

plt.xlabel('Codebase Size')  # 设置横轴标签
plt.ylabel('Control Flow Count')  # 设置纵轴标签
plt.title('Codebase Size vs Control Flow Count')  # 设置图表标题

plt.grid(False)  # 关闭网格线

plt.show()
