#!/usr/bin/env python3
"""
Session 152 (5000): BOOTSTRAPPING — S153 CONVENTION
====================================================
Uses ABSOLUTE percentile offsets (matching S153).
Tests: coverage, MAE, RMSE, parameter, offset, and ĝ(S) stability.
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import stats
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 150

# ========================================
# CONFIG
# ========================================
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 42

# S150 v4 parameters
POPT_MEAN = [-15322.010971, -0.000179, 15323.161698]
POPT_H04  = [1.293521, 4.209902, 0.000000]
POPT_H96  = [1.035665, 4.083053, 0.000000]
A_ALIGNED = 155.616
A_OPPOSED = 200.359

# ========================================
# FUNCTIONS
# ========================================
def power_law_mean(x, a, b, c):
    return a * (x + 1) ** (-b) + c

def h_quantile(S, a, b, c):
    return np.abs(a) * (S + 1) ** (-np.abs(b)) + np.abs(c)

def coverage_stats(actual, lo, hi):
    inside = (actual >= lo) & (actual <= hi)
    n = len(actual)
    cov = inside.mean()
    z = stats.norm.ppf(0.975)
    ci_lo = max(0, cov - z * np.sqrt(cov * (1 - cov) / n))
    ci_hi = min(1, cov + z * np.sqrt(cov * (1 - cov) / n))
    return cov, ci_lo, ci_hi

def compute_predictions(data, popt_m, o04_a, o96_a, o04_o, o96_o):
    """S153 convention: ABSOLUTE percentile offsets."""
    S = data['total_weighted'].values
    sign = data['s'].values
    state = data['align_state'].values

    g_pred = power_law_mean(S, *popt_m)
    h04 = h_quantile(S, *POPT_H04)
    h96 = h_quantile(S, *POPT_H96)

    # S153 convention: absolute percentiles as offsets
    q04 = np.where(state == 'aligned', -h04 + o04_a, -h04 + o04_o)
    q96 = np.where(state == 'aligned',  h96 + o96_a,  h96 + o96_o)
    A = np.where(state == 'aligned', A_ALIGNED, A_OPPOSED)

    delta_pred = sign * g_pred * S * A
    delta_raw_lo = sign * (g_pred + q04) * S * A
    delta_raw_hi = sign * (g_pred + q96) * S * A
    delta_lo = np.minimum(delta_raw_lo, delta_raw_hi)
    delta_hi = np.maximum(delta_raw_lo, delta_raw_hi)

    return delta_pred, delta_lo, delta_hi

# ========================================
# LOAD DATA
# ========================================
print("=" * 70)
print("SESSION 152 (S153 CONVENTION): BOOTSTRAPPING")
print("=" * 70)

df = pd.read_csv(r'C:\Users\k5d-q\OneDrive\Python\Zeta_PR\RELEVANT\PLAN C\SESSION19\zeta_features_5000.csv')

df['s'] = np.sign(df['delta_actual'])
df['R_aligned'] = df['s'] * df['R_prime']
df['align_state'] = df['R_aligned'].apply(lambda x: 'aligned' if x > 0 else 'opposed')

df['g_mirror_abs'] = (df['delta_actual'] / (df['total_weighted'] + 1e-8)).abs()
df.loc[df['align_state'] == 'aligned', 'g_norm'] = df['g_mirror_abs'] / A_ALIGNED
df.loc[df['align_state'] == 'opposed', 'g_norm'] = df['g_mirror_abs'] / A_OPPOSED

cal = df[df['set'] == 'CAL'].copy()
val = df[df['set'] == 'VAL'].copy()
print(f"  CAL: {len(cal)} | VAL: {len(val)}")

# ========================================
# BASELINE (S153 convention: absolute percentiles)
# ========================================
print("\n  Computing baseline (S153 convention)...")

g_pred_cal_base = power_law_mean(cal['total_weighted'].values, *POPT_MEAN)
resid_cal_base = cal['g_norm'].values - g_pred_cal_base

alig_mask_cal = cal['align_state'].values == 'aligned'
opp_mask_cal = cal['align_state'].values == 'opposed'

# S153 convention: ABSOLUTE percentiles
o04_a_base = np.percentile(resid_cal_base[alig_mask_cal], 4)
o96_a_base = np.percentile(resid_cal_base[alig_mask_cal], 96)
o04_o_base = np.percentile(resid_cal_base[opp_mask_cal], 4)
o96_o_base = np.percentile(resid_cal_base[opp_mask_cal], 96)

print(f"    Offsets: o04_alig={o04_a_base:+.6f}, o96_alig={o96_a_base:+.6f}")
print(f"             o04_opp ={o04_o_base:+.6f}, o96_opp ={o96_o_base:+.6f}")

# Baseline predictions on FIXED VAL
delta_pred_base, delta_lo_base, delta_hi_base = compute_predictions(
    val, POPT_MEAN, o04_a_base, o96_a_base, o04_o_base, o96_o_base
)

val_alig_mask = val['align_state'].values == 'aligned'
val_opp_mask = val['align_state'].values == 'opposed'

cov_alig_base, _, _ = coverage_stats(
    val['delta_actual'].values[val_alig_mask],
    delta_lo_base[val_alig_mask], delta_hi_base[val_alig_mask]
)
cov_opp_base, _, _ = coverage_stats(
    val['delta_actual'].values[val_opp_mask],
    delta_lo_base[val_opp_mask], delta_hi_base[val_opp_mask]
)

resid_base = val['delta_actual'].values - delta_pred_base
mae_base = np.abs(resid_base).mean()
rmse_base = np.sqrt(np.mean(resid_base ** 2))

print(f"\n  Baseline Results (VAL fixed):")
print(f"    Coverage Aligned: {cov_alig_base*100:.1f}%")
print(f"    Coverage Opposed: {cov_opp_base*100:.1f}%")
print(f"    MAE: {mae_base:.4f} μm")
print(f"    RMSE: {rmse_base:.4f} μm")

# ========================================
# BOOTSTRAP LOOP
# ========================================
print(f"\n{'='*70}")
print(f"RUNNING {N_BOOTSTRAP} BOOTSTRAP ITERATIONS (S153 convention)")
print(f"{'='*70}\n")

rng = np.random.default_rng(BOOTSTRAP_SEED)

cal_alig_idx = cal[cal['align_state'] == 'aligned'].index.to_numpy()
cal_opp_idx = cal[cal['align_state'] == 'opposed'].index.to_numpy()

bootstrap_records = []

for i in tqdm(range(N_BOOTSTRAP), desc="  Bootstrap", ncols=70):
    # Step 1: Resample CAL only (stratified, with replacement)
    boot_cal_idx = np.concatenate([
        rng.choice(cal_alig_idx, size=len(cal_alig_idx), replace=True),
        rng.choice(cal_opp_idx, size=len(cal_opp_idx), replace=True)
    ])
    cal_boot = cal.loc[boot_cal_idx]

    # Step 2: Refit mean function only
    try:
        popt_m, _ = curve_fit(
            power_law_mean,
            cal_boot['total_weighted'],
            cal_boot['g_norm'],
            p0=POPT_MEAN, maxfev=10000,
            bounds=([-np.inf, -1, -np.inf], [np.inf, 1, np.inf])
        )
    except Exception:
        popt_m = POPT_MEAN

    a_fit, b_fit, c_fit = popt_m

    # Step 3: Recompute ABSOLUTE percentile offsets (S153 convention)
    g_pred_cal_boot = power_law_mean(cal_boot['total_weighted'].values, *popt_m)
    resid_cal_boot = cal_boot['g_norm'].values - g_pred_cal_boot

    boot_alig = cal_boot['align_state'].values == 'aligned'
    boot_opp = cal_boot['align_state'].values == 'opposed'

    # ABSOLUTE percentiles (NOT deviations!)
    o04_a = np.percentile(resid_cal_boot[boot_alig], 4)
    o96_a = np.percentile(resid_cal_boot[boot_alig], 96)
    o04_o = np.percentile(resid_cal_boot[boot_opp], 4)
    o96_o = np.percentile(resid_cal_boot[boot_opp], 96)

    # Step 4: Evaluate on FIXED VAL (NOT resampled!)
    delta_pred, delta_lo, delta_hi = compute_predictions(
        val, popt_m, o04_a, o96_a, o04_o, o96_o
    )

    # Step 5: Record ALL metrics
    cov_a, _, _ = coverage_stats(
        val['delta_actual'].values[val_alig_mask],
        delta_lo[val_alig_mask], delta_hi[val_alig_mask]
    )
    cov_o, _, _ = coverage_stats(
        val['delta_actual'].values[val_opp_mask],
        delta_lo[val_opp_mask], delta_hi[val_opp_mask]
    )

    resid = val['delta_actual'].values - delta_pred
    mae = np.abs(resid).mean()
    rmse = np.sqrt(np.mean(resid ** 2))

    # Predicted function at representative S values
    g_at_1 = power_law_mean(np.array([1.0]), *popt_m)[0]
    g_at_10 = power_law_mean(np.array([10.0]), *popt_m)[0]
    g_at_100 = power_law_mean(np.array([100.0]), *popt_m)[0]

    bootstrap_records.append({
        'iteration': i + 1,
        'cov_aligned': cov_a,
        'cov_opposed': cov_o,
        'mae': mae,
        'rmse': rmse,
        'param_a': a_fit,
        'param_b': b_fit,
        'param_c': c_fit,
        'offset_04_aligned': o04_a,
        'offset_96_aligned': o96_a,
        'offset_04_opposed': o04_o,
        'offset_96_opposed': o96_o,
        'g_hat_S1': g_at_1,
        'g_hat_S10': g_at_10,
        'g_hat_S100': g_at_100,
    })

# ========================================
# SAVE CSV
# ========================================
bs = pd.DataFrame(bootstrap_records)
bs.to_csv('session152_bootstrap_final_results_5000.csv', index=False)
print(f"\n  ✅ SAVED: session152_bootstrap_final_results_5000.csv")
print(f"     ({len(bs)} rows, {len(bs.columns)} columns)")

# ========================================
# SUMMARY
# ========================================
print(f"\n{'='*70}")
print("RESULTS SUMMARY")
print(f"{'='*70}")

def print_metric(name, values, baseline, unit=''):
    mean = np.mean(values)
    std = np.std(values)
    p025 = np.percentile(values, 2.5)
    p975 = np.percentile(values, 97.5)
    cv = 100 * std / abs(mean) if abs(mean) > 1e-10 else 0
    print(f"  {name:25s}: {mean:.4f} ± {std:.4f} [{p025:.4f}, {p975:.4f}] "
          f"(CV={cv:.1f}% | Base={baseline:.4f})")
    return mean, std, cv

print(f"\n  --- Coverage (as %) ---")
cov_alig_mean, cov_alig_std, _ = print_metric("Aligned", bs['cov_aligned']*100, cov_alig_base*100)
cov_opp_mean, cov_opp_std, _ = print_metric("Opposed", bs['cov_opposed']*100, cov_opp_base*100)

print(f"\n  --- Error Metrics (μm) ---")
mae_mean, mae_std, mae_cv = print_metric("MAE", bs['mae'], mae_base)
rmse_mean, rmse_std, rmse_cv = print_metric("RMSE", bs['rmse'], rmse_base)

print(f"\n  --- Mean Function Parameters ---")
print_metric("a", bs['param_a'], POPT_MEAN[0])
print_metric("b", bs['param_b'], POPT_MEAN[1])
print_metric("c", bs['param_c'], POPT_MEAN[2])

print(f"\n  --- Predicted Function ĝ(S) ---")
print_metric("ĝ(S=1)", bs['g_hat_S1'], power_law_mean(np.array([1.0]), *POPT_MEAN)[0])
print_metric("ĝ(S=10)", bs['g_hat_S10'], power_law_mean(np.array([10.0]), *POPT_MEAN)[0])
print_metric("ĝ(S=100)", bs['g_hat_S100'], power_law_mean(np.array([100.0]), *POPT_MEAN)[0])

print(f"\n  --- Offsets (absolute percentiles) ---")
print_metric("o04_aligned", bs['offset_04_aligned'], o04_a_base)
print_metric("o96_aligned", bs['offset_96_aligned'], o96_a_base)
print_metric("o04_opposed", bs['offset_04_opposed'], o04_o_base)
print_metric("o96_opposed", bs['offset_96_opposed'], o96_o_base)

# Convergence
first_500 = bs.iloc[:500]
last_500 = bs.iloc[500:]
print(f"\n  --- Convergence (first 500 vs last 500) ---")
print(f"    Cov Aligned: {first_500['cov_aligned'].mean()*100:.2f}% → {last_500['cov_aligned'].mean()*100:.2f}%")
print(f"    Cov Opposed: {first_500['cov_opposed'].mean()*100:.2f}% → {last_500['cov_opposed'].mean()*100:.2f}%")

# ========================================
# VERDICT
# ========================================
print(f"\n{'='*70}")
print("VERDICT")
print(f"{'='*70}")

if cov_alig_mean >= 90 and cov_opp_mean >= 90:
    verdict = "VALIDATED FOR DEPLOYMENT"
elif cov_alig_mean >= 88 and cov_opp_mean >= 88:
    verdict = "VALIDATED WITH MONITORING"
else:
    verdict = "NEEDS REVIEW"

print(f"\n  {verdict}")
print(f"    Coverage >= 90%: Aligned={cov_alig_mean>=90} ({cov_alig_mean:.1f}%), Opposed={cov_opp_mean>=90} ({cov_opp_mean:.1f}%)")
print(f"    Coverage SD: {max(cov_alig_std, cov_opp_std):.2f}pp")

# ========================================
# VISUALIZATION
# ========================================
print(f"\n  Generating visualization...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

c_alig = '#2ecc71'
c_opp = '#e74c3c'
c_purple = '#6d4aff'

# Row 1
ax = axes[0, 0]
ax.hist(bs['cov_aligned']*100, bins=30, color=c_alig, alpha=0.6, label=f'Aligned ({cov_alig_mean:.1f}%±{cov_alig_std:.1f}pp)')
ax.hist(bs['cov_opposed']*100, bins=30, color=c_opp, alpha=0.6, label=f'Opposed ({cov_opp_mean:.1f}%±{cov_opp_std:.1f}pp)')
ax.axvline(90, color='gray', ls='--', lw=2, alpha=0.5, label='Target 90%')
ax.set_xlabel('Coverage (%)')
ax.set_ylabel('Count')
ax.set_title('(a) Coverage Distribution', fontweight='bold')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[0, 1]
ax.scatter(bs['mae'], bs['rmse'], c=range(len(bs)), cmap='viridis', s=8, alpha=0.5)
ax.axhline(rmse_base, color='#f39c12', ls='--', lw=2, alpha=0.7, label=f'Baseline ({rmse_base:.2f})')
ax.set_xlabel('MAE (μm)'); ax.set_ylabel('RMSE (μm)')
ax.set_title('(b) Error Stability', fontweight='bold')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[0, 2]
ax.hist(bs['g_hat_S1'], bins=30, color=c_purple, alpha=0.6, edgecolor='black')
base_g1 = power_law_mean(np.array([1.0]), *POPT_MEAN)[0]
ax.axvline(base_g1, color='orange', ls='--', lw=2, label=f'Baseline ({base_g1:.4f})')
ax.set_xlabel('ĝ(S=1)'); ax.set_ylabel('Count')
ax.set_title('(c) ĝ(S=1) Distribution', fontweight='bold')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Row 2
ax = axes[1, 0]
ax.hist(bs['param_b'], bins=30, color='#3498db', alpha=0.6, edgecolor='black')
ax.axvline(POPT_MEAN[1], color='orange', ls='--', lw=2, label=f'Baseline ({POPT_MEAN[1]:.6f})')
ax.set_xlabel('Parameter b'); ax.set_ylabel('Count')
ax.set_title('(d) Parameter b', fontweight='bold')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1, 1]
ax.hist(bs['offset_04_aligned'], bins=30, color=c_alig, alpha=0.5, label='o04 aligned')
ax.hist(bs['offset_96_aligned'], bins=30, color=c_opp, alpha=0.5, label='o96 aligned')
ax.set_xlabel('Offset Value'); ax.set_ylabel('Count')
ax.set_title('(e) Aligned Offsets', fontweight='bold')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1, 2]
ax.hist(bs['offset_04_opposed'], bins=30, color=c_alig, alpha=0.5, label='o04 opposed')
ax.hist(bs['offset_96_opposed'], bins=30, color=c_opp, alpha=0.5, label='o96 opposed')
ax.set_xlabel('Offset Value'); ax.set_ylabel('Count')
ax.set_title('(f) Opposed Offsets', fontweight='bold')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.suptitle(f'Session 152 (S153 Convention): Bootstrap Robustness\n'
             f'{N_BOOTSTRAP} iterations | VAL fixed | {verdict}',
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('session152_bootstrap_s153_convention.png', dpi=200, bbox_inches='tight')
plt.savefig('session152_bootstrap_s153_convention.pdf', bbox_inches='tight')
plt.show()

print(f"\n  ✅ Saved: session152_bootstrap_s153_convention.png/pdf")

# ========================================
# JSON SUMMARY
# ========================================
summary = {
    'session': 152,
    'version': 's153_convention',
    'n_bootstrap': N_BOOTSTRAP,
    'val_resampled': False,
    'offset_convention': 'absolute_percentiles',
    'verdict': verdict,
    'baseline': {
        'cov_aligned': cov_alig_base,
        'cov_opposed': cov_opp_base,
        'mae': mae_base,
        'rmse': rmse_base,
    },
    'bootstrap': {
        'cov_aligned': {'mean': cov_alig_mean, 'std': cov_alig_std},
        'cov_opposed': {'mean': cov_opp_mean, 'std': cov_opp_std},
        'mae': {'mean': mae_mean, 'std': mae_std, 'cv': mae_cv},
        'rmse': {'mean': rmse_mean, 'std': rmse_std, 'cv': rmse_cv},
    },
}

with open('session152_summary_s153_convention.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n{'='*70}")
print(f"SESSION 152 (S153 CONVENTION) COMPLETE! 🎉🐍")
print(f"{'='*70}")