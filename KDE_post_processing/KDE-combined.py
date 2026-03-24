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
output_path = os.path.join(script_dir, "ell_cb.png")
data_path_0 = sys.argv[1]
data_path_1 = sys.argv[2]
data_path_2 = sys.argv[3]
#/media/yuning/新加卷/Downloads/pyring/outputs/gw150914_EsGB_1024_example-2/Nested_sampler/posterior.dat

data_0 = np.loadtxt(data_path_0)
data_1 = np.loadtxt(data_path_1)
data_2 = np.loadtxt(data_path_2)
ell_0 = data_0[:,7]
ell_1 = data_1[:,11]
ell_2 = data_2[:,15]  #5th in .dat, python count from 0, so 4

ell_reflect_0 = np.concatenate([ell_0,-ell_0])
ell_reflect_1 = np.concatenate([ell_1,-ell_1])
ell_reflect_2 = np.concatenate([ell_2,-ell_2])
#kde_0 = gaussian_kde(ell_reflect_0, bw_method='scott')
#kde_1 = gaussian_kde(ell_reflect_1, bw_method='scott')
#kde_2 = gaussian_kde(ell_reflect_2, bw_method='scott')
kde_0 = gaussian_kde(ell_reflect_0, bw_method=0.08)
kde_1 = gaussian_kde(ell_reflect_1, bw_method=0.08)
kde_2 = gaussian_kde(ell_reflect_2, bw_method=0.08)
x_0 = np.linspace(min(ell_0), max(ell_0), 500)
x_1 = np.linspace(min(ell_1), max(ell_1), 500)
x_2 = np.linspace(min(ell_2), max(ell_2), 500)
pdf_0 = kde_0(x_0) * 2
pdf_1 = kde_1(x_1) * 2
pdf_2 = kde_2(x_2) * 2

peak_0 = x_0[np.argmax(pdf_0)]
peak_1 = x_1[np.argmax(pdf_1)]
peak_2 = x_2[np.argmax(pdf_2)]
upper_90_0 = np.percentile(ell_0, 90)
upper_90_1 = np.percentile(ell_1, 90)
upper_90_2 = np.percentile(ell_2, 90)

y_u_0 = kde_0(upper_90_0) * 2
y_u_1 = kde_1(upper_90_1) * 2
y_u_2 = kde_2(upper_90_2) * 2


plt.figure(figsize=(7,4))

plt.hist(ell_0, bins=20, density=True, histtype='step', color='tab:green', linestyle="--", alpha=0.7, label="$N_{max}$ = 0: posterior data")
plt.plot(x_0, pdf_0, linewidth=2, color='tab:green', label="KDE")
plt.vlines(upper_90_0, ymin=0, ymax=y_u_0, linestyle="-.", color='tab:green', label=f"90% UCI: $\ell$ $\sim$ {upper_90_0:.4f} km")

plt.hist(ell_1, bins=20, density=True, histtype='step', color='mediumpurple', linestyle="--", alpha=0.7, label="$N_{max}$ = 1: posterior data")
plt.plot(x_1, pdf_1, linewidth=2, color='mediumpurple', label="KDE")
plt.vlines(upper_90_1, ymin=0, ymax=y_u_1, linestyle="-.", color='mediumpurple', label=f"90% UCI: $\ell$ $\sim$ {upper_90_1:.4f} km")

plt.hist(ell_2, bins=20, density=True, histtype='step', color='tab:blue', linestyle="--", alpha=0.7, label="$N_{max}$ = 2: posterior data")
plt.plot(x_2, pdf_2, linewidth=2, color='tab:blue', label="KDE")
plt.vlines(upper_90_2, ymin=0, ymax=y_u_2, linestyle="-.", color='tab:blue', label=f"90% UCI: $\ell$ $\sim$ {upper_90_2:.4f} km")

plt.xlim(left=0)
plt.xlabel("$\ell$ (km)")
plt.ylabel("Probability Density")
plt.title("GW250114 - Quartic_3 EFT")
plt.legend()
#plt.legend(loc='upper left')

plt.tight_layout()
plt.savefig(output_path, dpi=300)

plt.show()

# 输出
print("Saved to:", output_path)
print("\n===== RESULTS =====")
print(f"Peak_0 (MAP): {peak_0:.6f}")
print(f"90% upper bound_0: [{upper_90_0:.6f}, {y_u_0[0]:.6f}]")
print(f"Peak_1 (MAP): {peak_1:.6f}")
print(f"90% upper bound_1: [{upper_90_1:.6f}, {y_u_1[0]:.6f}]")
print(f"Peak_2 (MAP): {peak_2:.6f}")
print(f"90% upper bound_2: [{upper_90_2:.6f}, {y_u_2[0]:.6f}]")


#how to run
#python3 KDE-combined.py /media/yuning/新加卷/Downloads/pyring/outputs/gw150914_EFT_q3_1024_example-2/Nested_sampler/posterior.dat /media/yuning/新加卷/Downloads/pyring/outputs/gw150914_EFT_q3_1024_example-1/Nested_sampler/posterior.dat /media/yuning/新加卷/Downloads/pyring/outputs/gw150914_EFT_q3_1024_example-0/Nested_sampler/posterior.dat
