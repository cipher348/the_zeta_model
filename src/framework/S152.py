#!/usr/bin/env python3
"""
Figure 3: Bootstrap Coverage Stability (S152)
==============================================
Publication-ready version of session152_bootstrap_s153_convention.png
Shows: Coverage histograms, error stability, parameter distributions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'figure.dpi': 150,
    'font.family': 'serif',
})

# ========================================
# LOAD BOOTSTRAP DATA
# ========================================
bs = pd.read_csv('session152_bootstrap_final_results_5000.csv')

# S152 summary stats (from real output)
COV_ALIG_MEAN = 91.5984
COV_ALIG_STD = 0.9414
COV_ALIG_CI_LO = 89.5485
COV_ALIG_CI_HI = 93.6455
COV_OPP_MEAN = 93.9366
COV_OPP_STD = 0.5774
COV_OPP_CI_LO = 92.4827
COV_OPP_CI_HI = 95.0153

MAE_MEAN = 2.0183
MAE_STD = 0.2310
RMSE_MEAN = 2.7874
RMSE_STD = 0.3303

G_S1_MEAN = -0.3080
G_S1_STD = 0.0817

PARAM_B_MEAN = 0.0009
PARAM_B_STD = 0.0316

O04_ALIG_MEAN = -0.6816
O04_ALIG_STD = 0.0742
O96_ALIG_MEAN = 2.1958
O96_ALIG_STD = 0.3915
O04_OPP_MEAN = -0.6905
O04_OPP_STD = 0.0746
O96_OPP_MEAN = 1.9688
O96_OPP_STD = 0.2053

# ========================================
# PLOT (2x3 GRID)
# ========================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

C_ALIG = '#2ecc71'
C_OPP = '#e74c3c'
C_PULPLE = '#6d4aff'

# --- ROW 1 ---
# (a) Coverage Histograms
ax = axes[0, 0]
ax.hist(bs['cov_aligned']*100, bins=30, color=C_ALIG, alpha=0.6,
        label=f'Aligned ({COV_ALIG_MEAN:.1f}%±{COV_ALIG_STD:.1f}pp)')
ax.hist(bs['cov_opposed']*100, bins=30, color=C_OPP, alpha=0.6,
        label=f'Opposed ({COV_OPP_MEAN:.1f}%±{COV_OPP_STD:.1f}pp)')
ax.axvline(90, color='gray', ls='--', lw=2, alpha=0.5, label='Target 90%')
ax.set_xlabel('Coverage (%)', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_title('(a) Coverage Distribution', fontweight='bold', fontsize=12)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# (b) Error Stability (MAE vs RMSE)
ax = axes[0, 1]
scatter = ax.scatter(bs['mae'], bs['rmse'], c=np.arange(len(bs)), cmap='viridis',
                     s=10, alpha=0.5, edgecolors='none')
ax.axhline(4.31, color='#f39c12', ls='--', lw=2, alpha=0.7,
           label=f'Baseline RMSE (4.31 µm)')
ax.axvline(3.11, color='#f39c12', ls=':', lw=2, alpha=0.7,
           label=f'Baseline MAE (3.11 µm)')
ax.set_xlabel('MAE (µm)', fontsize=11)
ax.set_ylabel('RMSE (µm)', fontsize=11)
ax.set_title('(b) Error Metric Stability', fontweight='bold', fontsize=12)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

corr = np.corrcoef(bs['mae'], bs['rmse'])[0, 1]
ax.text(0.05, 0.95, f'r = {corr:.4f}', transform=ax.transAxes, fontsize=9,
        bbox=dict(facecolor='white', alpha=0.8))

# (c) g_hat(S=1) Distribution
ax = axes[0, 2]
ax.hist(bs['g_hat_S1'], bins=30, color=C_PULPLE, alpha=0.6, edgecolor='black')
ax.axvline(-0.7504, color='orange', ls='--', lw=2,
           label=f'Baseline (-0.750)')
ax.set_xlabel("$\\hat{g}(S{=}1)$", fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_title('(c) Predicted Function $\\hat{g}(S{=}1)$', fontweight='bold', fontsize=12)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# --- ROW 2 ---
# (d) Parameter b Distribution
ax = axes[1, 0]
ax.hist(bs['param_b'], bins=30, color='#3498db', alpha=0.6, edgecolor='black')
ax.axvline(-0.000179, color='orange', ls='--', lw=2,
           label=f'Baseline (-0.000179)')
ax.set_xlabel('Parameter b', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_title('(d) Parameter b Distribution', fontweight='bold', fontsize=12)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# (e) Aligned Offsets
ax = axes[1, 1]
ax.hist(bs['offset_04_aligned'], bins=30, color=C_ALIG, alpha=0.5, label='$o_{04}$ aligned')
ax.hist(bs['offset_96_aligned'], bins=30, color=C_OPP, alpha=0.5, label='$o_{96}$ aligned')
ax.set_xlabel('Offset Value', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_title('(e) Aligned Offsets', fontweight='bold', fontsize=12)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# (f) Opposed Offsets
ax = axes[1, 2]
ax.hist(bs['offset_04_opposed'], bins=30, color=C_ALIG, alpha=0.5, label='$o_{04}$ opposed')
ax.hist(bs['offset_96_opposed'], bins=30, color=C_OPP, alpha=0.5, label='$o_{96}$ opposed')
ax.set_xlabel('Offset Value', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_title('(f) Opposed Offsets', fontweight='bold', fontsize=12)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# --- MAIN TITLE ---
plt.suptitle('Figure 3: Bootstrap Coverage Stability (S152)\n'
             '1,000 iterations | VAL fixed | VALIDATED FOR DEPLOYMENT\n'
             f'Coverage: Aligned {COV_ALIG_MEAN:.1f}%±{COV_ALIG_STD:.1f}pp, '
             f'Opposed {COV_OPP_MEAN:.1f}%±{COV_OPP_STD:.1f}pp',
             fontsize=14, fontweight='bold', y=0.995)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('figure3_bootstrap_s152.png', dpi=200, bbox_inches='tight')
plt.savefig('figure3_bootstrap_s152.pdf', bbox_inches='tight')
plt.show()

print("Saved: figure3_bootstrap_s152.png/pdf")
