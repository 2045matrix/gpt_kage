"""
协方差和相关系数的图表演示
根据 covariance.md 的内容创建所有可视化图表
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from scipy.stats import pearsonr

# 设置中文字体
font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
custom_font = fm.FontProperties(fname=font_path)
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# 1. 正相关与负相关的示意图
# ============================================================================
def plot_positive_negative_correlation():
    """绘制正相关和负相关的概念图"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 正相关：城镇化率 vs 房价
    urbanization = np.array([30, 40, 50, 60, 70, 80])
    house_price = np.array([100, 150, 200, 280, 350, 450])
    
    axes[0].scatter(urbanization, house_price, s=100, color='red', alpha=0.6, edgecolors='darkred')
    z = np.polyfit(urbanization, house_price, 1)
    p = np.poly1d(z)
    axes[0].plot(urbanization, p(urbanization), "r--", linewidth=2, label='拟合直线')
    axes[0].set_xlabel('城镇化率 (%)', fontproperties=custom_font, fontsize=12)
    axes[0].set_ylabel('房价 (万元/m²)', fontproperties=custom_font, fontsize=12)
    axes[0].set_title('正相关：城镇化率 vs 房价', fontproperties=custom_font, fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=10)
    
    # 负相关：城镇化率 vs 出生率
    birth_rate = np.array([35, 28, 22, 16, 12, 8])
    
    axes[1].scatter(urbanization, birth_rate, s=100, color='blue', alpha=0.6, edgecolors='darkblue')
    z = np.polyfit(urbanization, birth_rate, 1)
    p = np.poly1d(z)
    axes[1].plot(urbanization, p(urbanization), "b--", linewidth=2, label='拟合直线')
    axes[1].set_xlabel('城镇化率 (%)', fontproperties=custom_font, fontsize=12)
    axes[1].set_ylabel('出生率 (‰)', fontproperties=custom_font, fontsize=12)
    axes[1].set_title('负相关：城镇化率 vs 出生率', fontproperties=custom_font, fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig('1_positive_negative_correlation.png', dpi=150, bbox_inches='tight')
    print("✓ 图表1已保存：1_positive_negative_correlation.png")
    plt.close()

# ============================================================================
# 2. 身高体重数据的散点图与矩形表示
# ============================================================================
def plot_rectangle_visualization():
    """绘制矩形面积表示相关性"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 身高体重数据
    heights = np.array([152, 160, 172, 175, 180])
    weights = np.array([45, 54, 44, 64, 80])
    mean_height = heights.mean()
    mean_weight = weights.mean()
    
    # 左图：所有数据点与矩形
    ax = axes[0]
    ax.scatter(heights, weights, s=150, color='black', zorder=5, edgecolors='black', linewidth=2)
    
    # 绘制均值线
    ax.axvline(mean_height, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axhline(mean_weight, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # 绘制矩形并着色
    for i in range(len(heights)):
        dx = heights[i] - mean_height
        dy = weights[i] - mean_weight
        product = dx * dy
        
        # 根据乘积符号选择颜色
        if product > 0:
            color = 'red'
            alpha = 0.3
        else:
            color = 'blue'
            alpha = 0.3
        
        # 绘制矩形
        from matplotlib.patches import Rectangle
        rect = Rectangle((mean_height, mean_weight), dx, dy, 
                        facecolor=color, edgecolor='black', 
                        alpha=alpha, linewidth=1.5)
        ax.add_patch(rect)
        
        # 标注数据点
        ax.text(heights[i], weights[i]+2, f'({heights[i]},{weights[i]})', 
               ha='center', fontsize=9, fontproperties=custom_font)
    
    ax.set_xlabel('身高 (cm)', fontproperties=custom_font, fontsize=12)
    ax.set_ylabel('体重 (kg)', fontproperties=custom_font, fontsize=12)
    ax.set_title('矩形面积表示相关性\n（红色=正相关，蓝色=负相关）', 
                fontproperties=custom_font, fontsize=13, fontweight='bold')
    ax.set_xlim(145, 190)
    ax.set_ylim(35, 90)
    ax.grid(True, alpha=0.3)
    
    # 右图：以均值点为原点
    ax = axes[1]
    centered_heights = heights - mean_height
    centered_weights = weights - mean_weight
    
    ax.scatter(centered_heights, centered_weights, s=150, color='black', 
              zorder=5, edgecolors='black', linewidth=2)
    
    # 绘制坐标轴（均值为原点）
    ax.axhline(0, color='black', linewidth=2)
    ax.axvline(0, color='black', linewidth=2)
    
    # 绘制象限标签
    ax.text(5, 15, 'Ⅰ象限\n(正相关)', fontsize=11, ha='center', 
           fontproperties=custom_font, color='red', fontweight='bold')
    ax.text(-8, 15, 'Ⅱ象限\n(负相关)', fontsize=11, ha='center', 
           fontproperties=custom_font, color='blue', fontweight='bold')
    ax.text(-8, -10, 'Ⅲ象限\n(正相关)', fontsize=11, ha='center', 
           fontproperties=custom_font, color='red', fontweight='bold')
    ax.text(5, -10, 'Ⅳ象限\n(负相关)', fontsize=11, ha='center', 
           fontproperties=custom_font, color='blue', fontweight='bold')
    
    # 绘制矩形
    for i in range(len(heights)):
        dx = centered_heights[i]
        dy = centered_weights[i]
        
        if dx * dy > 0:
            color = 'red'
            alpha = 0.3
        else:
            color = 'blue'
            alpha = 0.3
        
        from matplotlib.patches import Rectangle
        rect = Rectangle((0, 0), dx, dy, facecolor=color, 
                        edgecolor='black', alpha=alpha, linewidth=1.5)
        ax.add_patch(rect)
    
    ax.set_xlabel('身高 - 均值 (cm)', fontproperties=custom_font, fontsize=12)
    ax.set_ylabel('体重 - 均值 (kg)', fontproperties=custom_font, fontsize=12)
    ax.set_title('以均值点为原点的象限图', fontproperties=custom_font, 
                fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-12, 10)
    ax.set_ylim(-16, 20)
    
    plt.tight_layout()
    plt.savefig('2_rectangle_visualization.png', dpi=150, bbox_inches='tight')
    print("✓ 图表2已保存：2_rectangle_visualization.png")
    plt.close()

# ============================================================================
# 3. 协方差计算与展示
# ============================================================================
def plot_covariance_explanation():
    """绘制协方差的计算过程"""
    heights = np.array([152, 160, 172, 175, 180])
    weights = np.array([45, 54, 44, 64, 80])
    
    mean_height = heights.mean()
    mean_weight = weights.mean()
    
    # 计算协方差
    centered_heights = heights - mean_height
    centered_weights = weights - mean_weight
    products = centered_heights * centered_weights
    covariance = products.mean()
    
    # 计算标准差
    std_height = heights.std()
    std_weight = weights.std()
    
    # 计算相关系数
    correlation = covariance / (std_height * std_weight)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 第一个子图：数据表
    ax = axes[0, 0]
    ax.axis('off')
    
    table_data = []
    table_data.append(['同学', '身高(cm)', '体重(kg)', '身高-均值', '体重-均值', '乘积'])
    for i in range(len(heights)):
        table_data.append([
            f'{i+1}号',
            f'{heights[i]}',
            f'{weights[i]}',
            f'{centered_heights[i]:.1f}',
            f'{centered_weights[i]:.1f}',
            f'{products[i]:.1f}'
        ])
    table_data.append(['均值', f'{mean_height:.1f}', f'{mean_weight:.1f}', '/', '/', f'{covariance:.2f}'])
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=[0.08, 0.15, 0.12, 0.15, 0.15, 0.12])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    for i in range(len(table_data)):
        if i == 0 or i == len(table_data) - 1:
            for j in range(len(table_data[0])):
                table[(i, j)].set_facecolor('#40466e')
                table[(i, j)].set_text_props(weight='bold', color='white')
        else:
            for j in range(len(table_data[0])):
                if j == len(table_data[0]) - 1:
                    table[(i, j)].set_facecolor('#ffcccc' if products[i-1] > 0 else '#ccccff')
    
    ax.set_title('协方差计算表', fontproperties=custom_font, fontsize=13, fontweight='bold', pad=20)
    
    # 第二个子图：公式说明
    ax = axes[0, 1]
    ax.axis('off')
    
    formula_text = f"""
协方差公式：
Cov(X,Y) = E[(X - μₓ)(Y - μᵧ)]

计算结果：
• 均值身高 μₓ = {mean_height:.2f} cm
• 均值体重 μᵧ = {mean_weight:.2f} kg
• 协方差 = {covariance:.2f} (cm·kg)

判断：
Cov(X,Y) = {covariance:.2f} > 0
⟹ 身高和体重正相关
"""
    
    ax.text(0.1, 0.9, formula_text, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
           fontproperties=custom_font)
    ax.set_title('协方差判断', fontproperties=custom_font, fontsize=13, fontweight='bold')
    
    # 第三个子图：相关系数计算
    ax = axes[1, 0]
    ax.axis('off')
    
    correlation_text = f"""
相关系数公式：
ρ = Cov(X,Y) / (σₓ · σᵧ)

计算结果：
• 身高标准差 σₓ = {std_height:.2f} cm
• 体重标准差 σᵧ = {std_weight:.2f} kg
• 相关系数 ρ = {correlation:.4f}

评估：
{correlation:.4f} ∈ [0, 1]
⟹ 身高和体重呈强正相关
"""
    
    ax.text(0.1, 0.9, correlation_text, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
           fontproperties=custom_font)
    ax.set_title('相关系数计算', fontproperties=custom_font, fontsize=13, fontweight='bold')
    
    # 第四个子图：散点图
    ax = axes[1, 1]
    ax.scatter(heights, weights, s=100, color='red', alpha=0.6, edgecolors='darkred', linewidth=2)
    z = np.polyfit(heights, weights, 1)
    p = np.poly1d(z)
    ax.plot(heights, p(heights), "r--", linewidth=2, label=f'拟合直线')
    ax.set_xlabel('身高 (cm)', fontproperties=custom_font, fontsize=11)
    ax.set_ylabel('体重 (kg)', fontproperties=custom_font, fontsize=11)
    ax.set_title(f'身高 vs 体重\nρ = {correlation:.4f}', fontproperties=custom_font, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig('3_covariance_calculation.png', dpi=150, bbox_inches='tight')
    print("✓ 图表3已保存：3_covariance_calculation.png")
    plt.close()

# ============================================================================
# 4. 不同类型的相关关系
# ============================================================================
def plot_correlation_types():
    """绘制不同相关系数的散点图"""
    np.random.seed(42)
    n = 100
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    # 强正相关 (ρ ≈ 0.9)
    X = np.random.randn(n)
    Y_strong_pos = 0.9 * X + 0.1 * np.random.randn(n)
    rho, _ = pearsonr(X, Y_strong_pos)
    axes[0].scatter(X, Y_strong_pos, alpha=0.5, s=50)
    axes[0].set_title(f'强正相关\nρ = {rho:.3f}', fontproperties=custom_font, fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # 弱正相关 (ρ ≈ 0.3)
    Y_weak_pos = 0.3 * X + 0.7 * np.random.randn(n)
    rho, _ = pearsonr(X, Y_weak_pos)
    axes[1].scatter(X, Y_weak_pos, alpha=0.5, s=50, color='orange')
    axes[1].set_title(f'弱正相关\nρ = {rho:.3f}', fontproperties=custom_font, fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # 不相关 (ρ ≈ 0)
    Y_uncor = np.random.randn(n)
    rho, _ = pearsonr(X, Y_uncor)
    axes[2].scatter(X, Y_uncor, alpha=0.5, s=50, color='gray')
    axes[2].set_title(f'不相关\nρ = {rho:.3f}', fontproperties=custom_font, fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    # 弱负相关 (ρ ≈ -0.3)
    Y_weak_neg = -0.3 * X + 0.7 * np.random.randn(n)
    rho, _ = pearsonr(X, Y_weak_neg)
    axes[3].scatter(X, Y_weak_neg, alpha=0.5, s=50, color='red')
    axes[3].set_title(f'弱负相关\nρ = {rho:.3f}', fontproperties=custom_font, fontsize=12, fontweight='bold')
    axes[3].grid(True, alpha=0.3)
    
    # 强负相关 (ρ ≈ -0.9)
    Y_strong_neg = -0.9 * X + 0.1 * np.random.randn(n)
    rho, _ = pearsonr(X, Y_strong_neg)
    axes[4].scatter(X, Y_strong_neg, alpha=0.5, s=50, color='darkred')
    axes[4].set_title(f'强负相关\nρ = {rho:.3f}', fontproperties=custom_font, fontsize=12, fontweight='bold')
    axes[4].grid(True, alpha=0.3)
    
    # 非线性关系 (ρ ≈ 0，但有明显的二次关系)
    X_nl = np.linspace(-3, 3, n)
    Y_nl = X_nl**2 + 0.5 * np.random.randn(n)
    rho, _ = pearsonr(X_nl, Y_nl)
    axes[5].scatter(X_nl, Y_nl, alpha=0.5, s=50, color='purple')
    axes[5].set_title(f'非线性关系\nρ = {rho:.3f} (但有二次关系)', 
                     fontproperties=custom_font, fontsize=12, fontweight='bold')
    axes[5].grid(True, alpha=0.3)
    
    for ax in axes:
        ax.set_xlabel('X', fontsize=10)
        ax.set_ylabel('Y', fontsize=10)
    
    fig.suptitle('不同类型的相关关系', fontproperties=custom_font, fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('4_correlation_types.png', dpi=150, bbox_inches='tight')
    print("✓ 图表4已保存：4_correlation_types.png")
    plt.close()

# ============================================================================
# 5. 股票组合例子
# ============================================================================
def plot_stock_portfolio():
    """绘制股票相关性与组合风险"""
    days = np.arange(1, 21)
    
    # 三支股票的收益率序列
    stock_blue = np.array([100, 101, 102, 101, 103, 104, 105, 104, 106, 107,
                          108, 107, 109, 110, 111, 112, 113, 114, 115, 116])
    stock_green = np.array([100, 101, 102, 101, 103, 104, 105, 104, 106, 107,
                           108, 107, 109, 110, 111, 112, 113, 114, 115, 116])
    stock_red = np.array([100, 99, 98, 99, 97, 96, 95, 96, 94, 93,
                         92, 93, 91, 90, 89, 88, 87, 86, 85, 84])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 第一个子图：三支股票走势
    axes[0, 0].plot(days, stock_blue, 'o-', color='blue', linewidth=2, markersize=6, label='蓝色股票')
    axes[0, 0].plot(days, stock_green, 's-', color='green', linewidth=2, markersize=6, label='绿色股票')
    axes[0, 0].plot(days, stock_red, '^-', color='red', linewidth=2, markersize=6, label='红色股票')
    axes[0, 0].set_xlabel('交易日', fontproperties=custom_font, fontsize=11)
    axes[0, 0].set_ylabel('股价', fontproperties=custom_font, fontsize=11)
    axes[0, 0].set_title('三支股票的价格走势', fontproperties=custom_font, fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 第二个子图：蓝色 vs 绿色（正相关）
    axes[0, 1].scatter(stock_blue, stock_green, s=100, alpha=0.6, color='purple', edgecolors='black')
    z = np.polyfit(stock_blue, stock_green, 1)
    p = np.poly1d(z)
    axes[0, 1].plot(stock_blue, p(stock_blue), "k--", linewidth=2)
    rho, _ = pearsonr(stock_blue, stock_green)
    axes[0, 1].set_xlabel('蓝色股票价格', fontproperties=custom_font, fontsize=11)
    axes[0, 1].set_ylabel('绿色股票价格', fontproperties=custom_font, fontsize=11)
    axes[0, 1].set_title(f'蓝色 vs 绿色（正相关）\nρ = {rho:.3f}', fontproperties=custom_font, fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 第三个子图：蓝色 vs 红色（负相关）
    axes[1, 0].scatter(stock_blue, stock_red, s=100, alpha=0.6, color='orange', edgecolors='black')
    z = np.polyfit(stock_blue, stock_red, 1)
    p = np.poly1d(z)
    axes[1, 0].plot(stock_blue, p(stock_blue), "k--", linewidth=2)
    rho, _ = pearsonr(stock_blue, stock_red)
    axes[1, 0].set_xlabel('蓝色股票价格', fontproperties=custom_font, fontsize=11)
    axes[1, 0].set_ylabel('红色股票价格', fontproperties=custom_font, fontsize=11)
    axes[1, 0].set_title(f'蓝色 vs 红色（负相关）\nρ = {rho:.3f}', fontproperties=custom_font, fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 第四个子图：组合表现
    portfolio_pos_corr = (stock_blue + stock_green) / 2  # 正相关组合
    portfolio_neg_corr = (stock_blue + stock_red) / 2    # 负相关组合
    
    axes[1, 1].plot(days, portfolio_pos_corr, 'o-', color='green', linewidth=2.5, markersize=6, label='正相关组合\n(蓝+绿)')
    axes[1, 1].plot(days, portfolio_neg_corr, 's-', color='blue', linewidth=2.5, markersize=6, label='负相关组合\n(蓝+红)')
    axes[1, 1].fill_between(days, portfolio_pos_corr.min(), portfolio_pos_corr.max(), alpha=0.1, color='green')
    axes[1, 1].fill_between(days, portfolio_neg_corr.min(), portfolio_neg_corr.max(), alpha=0.1, color='blue')
    axes[1, 1].set_xlabel('交易日', fontproperties=custom_font, fontsize=11)
    axes[1, 1].set_ylabel('组合价值', fontproperties=custom_font, fontsize=11)
    axes[1, 1].set_title('组合风险对比\n（负相关组合波动更小）', fontproperties=custom_font, fontsize=12, fontweight='bold')
    axes[1, 1].legend(fontsize=10, loc='upper left')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('5_stock_portfolio.png', dpi=150, bbox_inches='tight')
    print("✓ 图表5已保存：5_stock_portfolio.png")
    plt.close()

# ============================================================================
# 6. 相关系数范围说明
# ============================================================================
def plot_correlation_scale():
    """绘制相关系数的范围和含义"""
    fig, ax = plt.subplots(figsize=(14, 2))
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.5, 1.5)
    
    # 绘制主轴
    ax.plot([-1, 1], [0.5, 0.5], 'k-', linewidth=3)
    
    # 标注点
    points = [-1, -0.7, -0.3, 0, 0.3, 0.7, 1]
    colors = ['darkred', 'red', 'orange', 'gray', 'lightgreen', 'green', 'darkgreen']
    labels = ['强负相关\n-1', '负相关\n-0.7', '弱负相关\n-0.3', '不相关\n0', '弱正相关\n0.3', '正相关\n0.7', '强正相关\n1']
    
    for point, color, label in zip(points, colors, labels):
        ax.plot(point, 0.5, 'o', markersize=15, color=color, markeredgecolor='black', markeredgewidth=2)
        ax.text(point, 0.1, label, ha='center', fontsize=10, fontproperties=custom_font, fontweight='bold')
    
    # 标注区域
    ax.fill_between([-1, -0.3], -0.2, 1.3, alpha=0.1, color='blue', label='负相关区间')
    ax.fill_between([-0.3, 0.3], -0.2, 1.3, alpha=0.1, color='gray', label='弱相关区间')
    ax.fill_between([0.3, 1], -0.2, 1.3, alpha=0.1, color='red', label='正相关区间')
    
    ax.set_xlabel('相关系数 ρ', fontproperties=custom_font, fontsize=13, fontweight='bold')
    ax.set_title('相关系数的范围和含义', fontproperties=custom_font, fontsize=14, fontweight='bold', pad=20)
    ax.set_yticks([])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.5), fontsize=11, ncol=3)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('6_correlation_scale.png', dpi=150, bbox_inches='tight')
    print("✓ 图表6已保存：6_correlation_scale.png")
    plt.close()

# ============================================================================
# 主函数
# ============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("协方差和相关系数的图表生成")
    print("=" * 60)
    print()
    
    print("生成图表中...")
    print()
    
    plot_positive_negative_correlation()
    plot_rectangle_visualization()
    plot_covariance_explanation()
    plot_correlation_types()
    plot_stock_portfolio()
    plot_correlation_scale()
    
    print()
    print("=" * 60)
    print("✓ 所有图表已生成完毕！")
    print("=" * 60)
    print()
    print("生成的图表文件：")
    print("  1. 1_positive_negative_correlation.png   - 正相关与负相关")
    print("  2. 2_rectangle_visualization.png         - 矩形面积表示法")
    print("  3. 3_covariance_calculation.png          - 协方差计算过程")
    print("  4. 4_correlation_types.png               - 不同相关类型")
    print("  5. 5_stock_portfolio.png                 - 股票组合例子")
    print("  6. 6_correlation_scale.png               - 相关系数范围")
    print()
