import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# 节点和分支结构
nodes = [
    ("021ed7a", "bag_words", "main"),
    ("2412f25", "init", "main"),
    ("efcd343", "done", "main/covariance"),
    ("a9abf55", "restore: 保留 bag-of-words", "main"),
]

# 画布设置
fig, ax = plt.subplots(figsize=(8, 4))
ax.set_xlim(-0.5, 3.5)
ax.set_ylim(-1, 2)
ax.axis('off')

# 坐标布局
positions = {
    "021ed7a": (0, 0),
    "2412f25": (1, 0),
    "efcd343": (2, 0),
    "a9abf55": (3, 0.5),
}

# 画主线
ax.plot([0, 1, 2], [0, 0, 0], color='black', linewidth=2, zorder=1)
# covariance 分支
ax.plot([2, 2.2], [0, 0.5], color='blue', linewidth=2, zorder=1)
# main 分支恢复提交
ax.plot([2, 3], [0, 0.5], color='black', linewidth=2, zorder=1)

# 画节点
for node, label, branch in nodes:
    x, y = positions[node]
    color = 'red' if branch == 'main' else 'blue'
    ax.scatter(x, y, s=400, color=color, edgecolor='black', zorder=2)
    ax.text(x, y+0.15, node, ha='center', fontsize=9, color='black')
    ax.text(x, y-0.18, label, ha='center', fontsize=10, color='black')
    if node == 'efcd343':
        ax.text(x+0.2, y+0.25, 'covariance', color='blue', fontsize=10)
    if node == 'a9abf55':
        ax.text(x, y+0.25, 'main', color='red', fontsize=10)

# 图例
main_patch = mpatches.Patch(color='red', label='main 分支')
cov_patch = mpatches.Patch(color='blue', label='covariance 分支')
ax.legend(handles=[main_patch, cov_patch], loc='upper left')

plt.title('gpt_kage 分支合并历史示意图', fontsize=13)
plt.tight_layout()
plt.savefig('branch_graph.png', dpi=150)
plt.close()
print('分支结构图已生成：branch_graph.png')
