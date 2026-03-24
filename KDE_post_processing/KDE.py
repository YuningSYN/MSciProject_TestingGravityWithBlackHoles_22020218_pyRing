#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import os
import sys

# =========================
# 路径处理
# =========================

script_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(script_dir, "ell.png")
data_path = sys.argv[1]
#/media/yuning/新加卷/Downloads/pyring/outputs/gw150914_EsGB_1024_example-2/Nested_sampler/posterior.dat

# =========================
# 读取数据
# =========================
data = np.loadtxt(data_path)
ell = data[:,4]  #5th in .dat, python count from 0, so 4

# =========================
# KDE
# =========================
ell_reflect = np.concatenate([ell,-ell])
kde = gaussian_kde(ell_reflect, bw_method='scott')
x = np.linspace(min(ell), max(ell), 500)
pdf = kde(x) * 2

# =========================
# 统计量
# =========================
peak = x[np.argmax(pdf)]
low_90 = np.percentile(ell,5)
high_90 = np.percentile(ell,95)
upper_90 = np.percentile(ell, 90)

y_h = kde(high_90) * 2
y_u = kde(upper_90) * 2

# =========================
# 作图
# =========================   ={peak:.3f}
plt.figure(figsize=(9,5))
plt.hist(ell, bins=25, density=True, alpha=0.3, label="Posterior data")
plt.plot(x, pdf, linewidth=2, label="KDE - ")
plt.xlim(left=0)

#plt.axvline(peak, linestyle='-', color='mediumseagreen', label=f"Peak")
#plt.axvline(low_90, linestyle="-.", color='mediumpurple', label="90% CI")
plt.vlines(high_90, ymin=0, ymax=y_h, linestyle="-.", color='mediumpurple', label="95% UCI")
plt.vlines(upper_90, ymin=0, ymax=y_u, linestyle="--", label="90% UCI")

plt.xlabel("ell")
plt.ylabel("Probability Density")
plt.title("GW150914")
plt.legend()

# 保存
plt.tight_layout()
plt.savefig(output_path, dpi=300)

plt.show()

# 输出
print("Saved to:", output_path)
print("\n===== RESULTS =====")
print(f"Peak (MAP): {peak:.6f}")
print(f"90% CI: [{high_90:.6f}, {y_h[0]:.6f}]")
print(f"90% upper bound: [{upper_90:.6f}, {y_u[0]:.6f}]")
