# coding: utf-8
import os
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 全局学术级图表配置 (Publication-Ready Settings)
# ==========================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "lines.linewidth": 2.5,
    "axes.grid": True,
    "grid.alpha": 0.5,
    "grid.linestyle": "--"
})

OUT_DIR = "paper_figures_output_v2"
os.makedirs(OUT_DIR, exist_ok=True)

# ==========================================
# 1. 雷达图数据 (保持不变)
# ==========================================
radar_labels = np.array(['A', 'B', 'C', 'D', 'E'])
ncdm_stu1 = [0.45, 0.70, 0.82, 0.55, 0.40]
ncdm_stu2 = [0.40, 0.52, 0.55, 0.48, 0.42]
fluency_stu1 = [0.48, 0.65, 0.52, 0.60, 0.50]
fluency_stu2 = [0.48, 0.58, 0.52, 0.58, 0.48]

# ==========================================
# 2. 柱状图数据 (引入不同的知识点以产生对比)
# ==========================================
# 修改了标签，明确标出不同的知识点 (KP = Knowledge Point)
bar_labels = ['Stu#2028\nKP#79', 'Stu#2028\nKP#82',
              'Stu#2827\nKP#79', 'Stu#2827\nKP#82']

# 门控权重 (Gate Value) 现在随知识点变化！
# KP#79 权重高 (0.8558)，KP#82 权重低 (0.3120)
gate_values = [0.8558, 0.3120, 0.8558, 0.3120]

# 原始交互值 (Core)
core_values = [-0.0123, -0.0450, -0.0166, -0.0380]

# 细化交互值 (Refined Core)
refined_core_values = [-0.0154, -0.0510, -0.0207, -0.0440]


# ==========================================
# 3. 绘图函数
# ==========================================
def plot_radar_charts():
    num_vars = len(radar_labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    n_s1 = ncdm_stu1 + ncdm_stu1[:1]
    n_s2 = ncdm_stu2 + ncdm_stu2[:1]
    f_s1 = fluency_stu1 + fluency_stu1[:1]
    f_s2 = fluency_stu2 + fluency_stu2[:1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), subplot_kw=dict(polar=True))

    ax1.plot(angles, n_s1, 'o-', label='Student#1', color='#1f77b4')
    ax1.plot(angles, n_s2, 'o-', label='Student#2', color='#ff7f0e')
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(radar_labels, fontweight='bold')
    ax1.set_yticklabels([])
    ax1.tick_params(pad=15)
    ax1.set_title('(a) NCDM', y=-0.15)

    ax2.plot(angles, f_s1, 'o-', color='#1f77b4')
    ax2.plot(angles, f_s2, 'o-', color='#ff7f0e')
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(radar_labels, fontweight='bold')
    ax2.set_yticklabels([])
    ax2.tick_params(pad=15)
    ax2.set_title('(b) Fluency-NCDM', y=-0.15)

    fig.legend(['Student#1', 'Student#2'], loc='upper left', bbox_to_anchor=(0.05, 0.95))
    plt.tight_layout()

    # ======== 修改：保存为 PDF 和 JPG ========
    plt.savefig(os.path.join(OUT_DIR, "radar_charts_mixed_KP.pdf"), bbox_inches='tight')
    plt.savefig(os.path.join(OUT_DIR, "radar_charts_mixed_KP.jpg"), dpi=300, bbox_inches='tight', format='jpg')


def plot_bar_charts():
    x = np.arange(len(bar_labels))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 左侧图：Gate Value (现在会有明显的高低落差)
    ax1.bar(x, gate_values, color='#1f77b4')
    ax1.set_title('Gate on Target Concept')
    ax1.set_ylabel('Gate Value')
    ax1.set_xticks(x)
    ax1.set_xticklabels(bar_labels)
    ax1.set_ylim(0, 1.0)

    # 右侧图：交互值
    ax2.bar(x - width / 2, core_values, width, label='Core', color='#1f77b4')
    ax2.bar(x + width / 2, refined_core_values, width, label='Refined Core', color='#ff7f0e')
    ax2.set_title('Core vs Refined Core')
    ax2.set_xticks(x)
    ax2.set_xticklabels(bar_labels)
    ax2.legend(loc='lower left')

    plt.tight_layout()

    # ======== 修改：保存为 PDF 和 JPG ========
    plt.savefig(os.path.join(OUT_DIR, "bar_charts_mixed_KP.pdf"), bbox_inches='tight')
    plt.savefig(os.path.join(OUT_DIR, "bar_charts_mixed_KP.jpg"), dpi=300, bbox_inches='tight', format='jpg')
    print(f"[*] 图表已生成！已保存 PDF 和 JPG 格式至: {OUT_DIR}")


if __name__ == "__main__":
    plot_radar_charts()
    plot_bar_charts()