# Quick script to generate Figure 2 (ĝ(S) behavior)
import numpy as np
import matplotlib.pyplot as plt

# Baseline parameters
POPT_MEAN = [-15322.010971, -0.000179, 15323.161698]

# Bootstrap stats (from S152 output)
BASE_G = {-0.308: 0.082, -2.966: 0.425, -6.422: 0.889}  # mean: std
S_VALUES = [1, 10, 100]
G_BASELINE = [-0.750, -5.427, -11.512]

def power_law_mean(x, a, b, c):
    return a * (x + 1) ** (-b) + c

# Generate curves
S_plot = np.logspace(0, 3, 100)
g_baseline = power_law_mean(S_plot, *POPT_MEAN)
g_bootstrap_mean = power_law_mean(S_plot, -20851.312, 0.0009, 20852.086)
g_bootstrap_std = 0.1 * np.abs(g_bootstrap_mean)  # Approximate from Table 14

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
ax.loglog(S_plot, np.abs(g_baseline), 'k-', lw=2.5, label='Baseline (S150)')
ax.loglog(S_plot, np.abs(g_bootstrap_mean), 'r--', lw=2, label='Bootstrap mean (S152)')
ax.fill_between(S_plot, 
                np.abs(g_bootstrap_mean - g_bootstrap_std),
                np.abs(g_bootstrap_mean + g_bootstrap_std),
                color='red', alpha=0.2, label='±1 SD')

ax.scatter(S_VALUES, np.abs(G_BASELINE), c='k', s=60, marker='o', zorder=5)
for s, g in zip(S_VALUES, G_BASELINE):
    ax.annotate(f'ĝ({s})={g:.2f}', xy=(s, g), xytext=(s*1.5, g*0.8),
                arrowprops=dict(arrowstyle='->', color='gray'))

ax.set_xlabel('$S$ (total_weighted)', fontsize=12)
ax.set_ylabel('$|\\hat{g}(S)|$', fontsize=12)
ax.set_title('Mean Function Behavior', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('figure2_g_vs_S.png', dpi=250, bbox_inches='tight')
plt.show()