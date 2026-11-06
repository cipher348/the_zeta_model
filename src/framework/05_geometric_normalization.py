#!/usr/bin/env python3
"""
SESSION 123 (COMPLETE FIXED): GEOMETRIC NORMALIZATION
======================================================
Following Cipher's critique:

  Δt ≈ a(t) · δ_∥ + b(t) · δ_⊥  (not just f(orientation))

PROTOCOL:
  1. Store: Δ_actual, R'(t), R''(t), arg D_N, tangent_angle
  2. First check: Is SIGN(Δ) predicted correctly?
  3. Then check separately: Can MAGNITUDE be calibrated?
  4. NO joint regression until sign and scale are separated!

ALL BUGS FROM PREVIOUS RUN FIXED:
  - correct_sign_mask properly defined
  - X_val_mag_scaled scoped correctly
  - Panel A uses subgridspec (not ax.subplot)
  - Negative magnitudes clamped to 0.01
  - next_step_msg defined in FAZIT
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import linregress
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)

# ============================================================
# PARAMETERS (FROZEN)
# ============================================================

ALPHA = 12.89
BETA  = 370.92
GAMMA = 0.4811
LAM_OPT = 112.4
PHI_OPT = 2.035

X_STAR = np.sqrt(ALPHA / BETA)

KNOWN_ZEROS = [14.1347, 21.0220, 25.0108, 30.4249, 32.9351,
               37.5862, 40.9187, 43.3271, 48.0052, 49.7738,
               52.9703, 56.4462, 59.3470, 60.8318, 65.1125,
               67.0798, 69.5464, 72.0672, 75.7047, 77.1448]

CALIBRATION_SET = KNOWN_ZEROS[:10]
VALIDATION_SET = KNOWN_ZEROS[10:]

# ============================================================
# CORE: DUFFING-ZETA WITH FULL GEOMETRY
# ============================================================

def duffing_zeta_with_full_geometry(s_complex, N=5000, h=0.01,
                                     lam=LAM_OPT, phi=PHI_OPT):
    sigma = s_complex.real
    t_val = s_complex.imag

    x_re, v_re = 0.0, 0.0
    x_im, v_im = 0.0, 0.0

    traj_x_re = np.zeros(N)
    traj_v_re = np.zeros(N)
    traj_x_im = np.zeros(N)
    traj_v_im = np.zeros(N)

    start_idx = int(N * 0.5)

    for n in range(N):
        log_n = np.log(n + 1)
        mag = np.exp(-sigma * log_n)

        f_re = mag * np.cos(t_val * log_n + phi)
        f_im = -mag * np.sin(t_val * log_n + phi)

        a_re = ALPHA * x_re - BETA * x_re**3 - GAMMA * v_re + lam * f_re
        v_re += h * a_re
        x_re += h * v_re
        traj_x_re[n] = x_re
        traj_v_re[n] = v_re

        a_im = ALPHA * x_im - BETA * x_im**3 - GAMMA * v_im + lam * f_im
        v_im += h * a_im
        x_im += h * v_im
        traj_x_im[n] = x_im
        traj_v_im[n] = v_im

    avg_re = np.mean(traj_x_re[start_idx:])
    avg_im = np.mean(traj_x_im[start_idx:])
    zeta_d = avg_re + 1j * avg_im

    # Extract geometry from terminal window
    win_start = start_idx + max(0, (N - start_idx) - 200)
    win_end = N

    win_x_re = traj_x_re[win_start:win_end]
    win_v_re = traj_v_re[win_start:win_end]
    win_x_im = traj_x_im[win_start:win_end]
    win_v_im = traj_v_im[win_start:win_end]

    # Slope and curvature of |R| near end
    mag_seq = np.sqrt(win_x_re**2 + win_x_im**2)
    if len(mag_seq) > 2:
        R_prime = np.gradient(mag_seq, h)
        R_double_prime = np.gradient(R_prime, h)
        slope_at_min = np.mean(R_prime[-50:])
        curvature_at_min = np.mean(R_double_prime[-50:])
    else:
        slope_at_min = 0.0
        curvature_at_min = 0.0

    complex_phase = np.angle(zeta_d)

    # Tangent angle in complex plane
    dx_re = np.mean(np.diff(win_x_re))
    dx_im = np.mean(np.diff(win_x_im))
    tangent_angle = np.arctan2(dx_im, dx_re)

    # Well-hopping
    well_positive = np.abs(win_x_re) > X_STAR * 0.8
    well_negative = np.abs(win_x_re) < X_STAR * 0.2
    hopping_frac = min(np.mean(well_positive), np.mean(well_negative))

    # Force-velocity sync
    log_n_array = np.log(np.arange(N) + 1)
    force_phase_tail = t_val * log_n_array[win_start:win_end] + phi
    force_sign = np.sign(np.cos(force_phase_tail[-50:]))
    vel_sign = np.sign(win_v_re[-50:])
    sync_index = np.mean(force_sign * vel_sign)

    geometry = {
        'slope_R_prime': slope_at_min,
        'curvature_R_double_prime': curvature_at_min,
        'complex_phase_arg_DN': complex_phase,
        'tangent_angle': tangent_angle,
        'hopping_frac': hopping_frac,
        'sync_index': sync_index,
        'response_magnitude': np.abs(zeta_d),
        'x_re_mean': np.mean(win_x_re),
        'x_im_mean': np.mean(win_x_im),
        'v_re_std': np.std(win_v_re),
        'v_im_std': np.std(win_v_im)
    }

    return zeta_d, geometry


# ============================================================
# MAIN
# ============================================================

print("=" * 80)
print("SESSION 123: GEOMETRIC NORMALIZATION & TWO-STAGE ANALYSIS")
print("=" * 80)

print(f"""
  CIPHER'S METHODOLOGY:
    1. Store: Δ_actual, R'(t), R''(t), arg D_N, tangent_angle
    2. Check: Is SIGN(Δ) predicted correctly?
    3. Check separately: Can MAGNITUDE be calibrated?
    4. NO joint regression until sign/scale separated!
""")

all_results = []

# ============================================================
# [1] SCAN ALL 20 ZEROS WITH GEOMETRY
# ============================================================

print("=" * 80)
print("[1] SCANNING ALL 20 ZEROS WITH GEOMETRY STORAGE")
print("=" * 80)

for i, kz in enumerate(KNOWN_ZEROS):
    print(f"\n  Zero #{i+1}: t ≈ {kz:.4f}...")

    t_local = np.linspace(max(0.1, kz - 2), kz + 2, 200)
    abs_vals = []

    for tv in t_local:
        s_val = 0.5 + 1j * tv
        zd, _ = duffing_zeta_with_full_geometry(s_val, N=5000)
        abs_vals.append(abs(zd))

    min_idx = np.argmin(abs_vals)
    t_detected = t_local[min_idx]
    delta_actual = t_detected - kz

    s_at_min = 0.5 + 1j * t_detected
    _, geometry = duffing_zeta_with_full_geometry(s_at_min, N=5000)

    result = {
        'zero_index': i + 1,
        't_known': kz,
        't_detected': t_detected,
        'delta_actual': delta_actual,
        'geometry': geometry,
        'is_calibration': i < 10
    }
    all_results.append(result)
    print(f"    t_detected = {t_detected:.4f}, Δ_actual = {delta_actual:+.4f}")
    print(f"    R'(t) = {geometry['slope_R_prime']:.4f}, R''(t) = {geometry['curvature_R_double_prime']:.4f}")
    print(f"    arg(D_N) = {geometry['complex_phase_arg_DN']:.4f}, tan_angle = {geometry['tangent_angle']:.4f}")

# ============================================================
# [2a] PHASE 1: SIGN PREDICTION
# ============================================================

print("\n" + "=" * 80)
print("[2a] PHASE 1: SIGN PREDICTION ACCURACY")
print("=" * 80)

geom_keys = ['slope_R_prime', 'curvature_R_double_prime', 'complex_phase_arg_DN',
             'tangent_angle', 'hopping_frac', 'sync_index']

sign_features = np.array([[r['geometry'][k] for k in geom_keys] for r in all_results])
sign_targets = np.array([np.sign(r['delta_actual']) for r in all_results])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(sign_features)

cal_mask = np.array([r['is_calibration'] for r in all_results])
val_mask = ~cal_mask

X_cal = X_scaled[cal_mask]
y_cal = sign_targets[cal_mask]
X_val = X_scaled[val_mask]
y_val = sign_targets[val_mask]

sign_classifier = LogisticRegression()
sign_classifier.fit(X_cal, y_cal)

sign_pred_cal = sign_classifier.predict(X_cal)
sign_pred_val = sign_classifier.predict(X_val)

acc_cal = np.mean(sign_pred_cal == y_cal)
acc_val = np.mean(sign_pred_val == y_val)

print(f"  Calibration set (zeros 1-10): Sign accuracy = {acc_cal*100:.1f}%")
print(f"  Validation set (zeros 11-20): Sign accuracy = {acc_val*100:.1f}%")

# --- FIX: Define correct_sign_mask BEFORE Phase 2b ---
combined_pred_signs = np.concatenate([sign_pred_cal, sign_pred_val])
correct_sign_mask = (sign_targets == combined_pred_signs)

val_offset = np.sum(cal_mask)

print(f"\n  SIGN PREDICTION RESULTS:")
print(f"    {'#':>4} {'t_known':>10} {'Δ_actual':>10} {'Actual':>8} {'Pred':>8} {'Match':>6}")
print("    " + "-" * 52)

for i, r in enumerate(all_results):
    if cal_mask[i]:
        pred_sign = sign_pred_cal[i]
    else:
        pred_sign = sign_pred_val[i - val_offset]
    actual_sign = int(np.sign(r['delta_actual']))
    match = "✓" if pred_sign == actual_sign else "✗"
    print(f"    {r['zero_index']:>4} {r['t_known']:>10.4f} {r['delta_actual']:>+10.4f} {actual_sign:>8} {int(pred_sign):>8} {match:>6}")

# ============================================================
# [2b] PHASE 2: MAGNITUDE CALIBRATION (correct-sign only)
# ============================================================

print("\n" + "=" * 80)
print("[2b] PHASE 2: MAGNITUDE CALIBRATION (ONLY IF SIGN CORRECT)")
print("=" * 80)

# --- FIX: Keep full-size arrays, filter with combined masks ---
mag_features_full = X_scaled  # Size 20
mag_targets_full = np.abs(np.array([r['delta_actual'] for r in all_results]))

cal_mag_mask = cal_mask & correct_sign_mask
val_mag_mask = val_mask & correct_sign_mask

X_cal_mag = mag_features_full[cal_mag_mask]
y_cal_mag = mag_targets_full[cal_mag_mask]
X_val_mag = mag_features_full[val_mag_mask]
y_val_mag = mag_targets_full[val_mag_mask]

print(f"  Correct-sign cases in calibration: {np.sum(cal_mag_mask)}/10")
print(f"  Correct-sign cases in validation:  {np.sum(val_mag_mask)}/10")

mae_cal = float('nan')
mae_val = float('nan')
y_pred_cal_mag = np.array([])
y_pred_val_mag = np.array([])

if len(y_cal_mag) > 0:
    mag_scaler = StandardScaler()
    X_cal_mag_scaled = mag_scaler.fit_transform(X_cal_mag)

    ridge_mag = Ridge(alpha=1.0)
    ridge_mag.fit(X_cal_mag_scaled, y_cal_mag)

    # --- FIX: Clamp negative predictions to 0.01 ---
    y_pred_cal_mag = np.maximum(ridge_mag.predict(X_cal_mag_scaled), 0.01)

    if len(X_val_mag) > 0:
        # --- FIX: Define X_val_mag_scaled BEFORE using it ---
        X_val_mag_scaled = mag_scaler.transform(X_val_mag)
        y_pred_val_mag = np.maximum(ridge_mag.predict(X_val_mag_scaled), 0.01)
        mae_val = np.mean(np.abs(y_val_mag - y_pred_val_mag))

    mae_cal = np.mean(np.abs(y_cal_mag - y_pred_cal_mag))

    print(f"\n  Magnitude calibration (correct-sign cases only):")
    print(f"    Calibration MAE: {mae_cal:.4f}")
    print(f"    Validation MAE:  {mae_val:.4f}" if not np.isnan(mae_val) else "    Validation MAE: N/A")

    print(f"\n  MAGNITUDE CALIBRATION RESULTS:")
    print(f"    {'#':>4} {'Set':>6} {'|Δ_actual|':>12} {'|Δ_pred|':>12} {'Ratio':>10}")
    print("    " + "-" * 48)

    cal_indices = np.where(cal_mag_mask)[0]
    val_indices = np.where(val_mag_mask)[0]

    for j, idx in enumerate(cal_indices):
        r = all_results[idx]
        ratio = np.abs(r['delta_actual']) / max(y_pred_cal_mag[j], 1e-10)
        print(f"    {r['zero_index']:>4} {'CAL':>6} {np.abs(r['delta_actual']):>12.4f} {y_pred_cal_mag[j]:>12.4f} {ratio:>10.2f}")

    for j, idx in enumerate(val_indices):
        r = all_results[idx]
        if j < len(y_pred_val_mag):
            ratio = np.abs(r['delta_actual']) / max(y_pred_val_mag[j], 1e-10)
            print(f"    {r['zero_index']:>4} {'VAL':>6} {np.abs(r['delta_actual']):>12.4f} {y_pred_val_mag[j]:>12.4f} {ratio:>10.2f}")
else:
    print("  WARNING: No calibration cases with correct sign prediction!")

# ============================================================
# [3] GEOMETRIC CORRELATIONS
# ============================================================

print("\n" + "=" * 80)
print("[3] GEOMETRIC CORRELATIONS: What drives the error?")
print("=" * 80)

deltas = np.array([r['delta_actual'] for r in all_results])

print("\n  Correlation matrix: Δ vs geometric features")
print("  " + "-" * 60)

correlations = {}
for key in geom_keys:
    geom_vals = np.array([r['geometry'][key] for r in all_results])
    corr_coef = np.corrcoef(geom_vals, deltas)[0, 1]
    correlations[key] = corr_coef
    strength = 'Strong' if abs(corr_coef) > 0.7 else 'Moderate' if abs(corr_coef) > 0.4 else 'Weak'
    print(f"    {key:>25}: r = {corr_coef:>+.4f} ({strength})")

# ============================================================
# VISUALIZATION (ALL BUGS FIXED)
# ============================================================

print("\n" + "=" * 80)
print("VISUALISIERUNG")
print("=" * 80)

fig = plt.figure(figsize=(24, 20))
gs = GridSpec(4, 4, figure=fig, hspace=0.45, wspace=0.35)

# --- Panel A: Feature Distributions (FIXED with subgridspec) ---
ax1 = fig.add_subplot(gs[0, 0:2])
ax1.set_axis_off()
ax1.set_title('A) Distribution of Key Geometric Features', fontsize=10, fontweight='bold')
sub_gs = gs[0, 0:2].subgridspec(2, 3, hspace=0.4, wspace=0.3)
for i, key in enumerate(geom_keys[:5]):
    ax_sub = fig.add_subplot(sub_gs[i // 3, i % 3])
    vals = np.array([r['geometry'][key] for r in all_results])
    ax_sub.hist(vals, bins=10, color='#6d4aff', alpha=0.7, edgecolor='k')
    ax_sub.set_title(key.replace('_', ' '), fontsize=7)
    ax_sub.tick_params(labelsize=6)
    ax_sub.grid(True, alpha=0.3)

# --- Panel B: Sign Prediction Results ---
ax2 = fig.add_subplot(gs[0, 2])
indices = range(1, 21)
actual_sigs = [int(np.sign(r['delta_actual'])) for r in all_results]
pred_sigs = []
for i, r in enumerate(all_results):
    if cal_mask[i]:
        pred_sigs.append(int(sign_pred_cal[i]))
    else:
        pred_sigs.append(int(sign_pred_val[i - val_offset]))

colors = ['#2ecc71' if a == p else '#e74c3c' for a, p in zip(actual_sigs, pred_sigs)]
bars = ax2.bar(indices, actual_sigs, color=colors, alpha=0.7, edgecolor='k')
ax2.axhline(0, color='black', lw=1.5)
ax2.set_xlabel('Zero Index', fontsize=9)
ax2.set_ylabel('Sign of Δ', fontsize=9)
ax2.set_title('B) Sign Prediction: Actual\n(Green=Match, Red=Mismatch)', fontsize=10, fontweight='bold')
ax2.set_ylim(-1.5, 1.5)
ax2.set_xticks(range(1, 21))
ax2.grid(True, alpha=0.3, axis='y')

# Vertical line between cal/val
ax2.axvline(10.5, color='gray', ls='--', lw=1.5, alpha=0.5)
ax2.text(5, 1.3, 'Calibration', fontsize=7, ha='center', color='blue')
ax2.text(15, 1.3, 'Validation', fontsize=7, ha='center', color='red')

# --- Panel C: Δ vs Slope R'(t) ---
ax3 = fig.add_subplot(gs[0, 3])
slopes = np.array([r['geometry']['slope_R_prime'] for r in all_results])
cal_mask_vis = cal_mask

ax3.scatter(slopes[cal_mask_vis], deltas[cal_mask_vis], color='blue', label='Calibration', alpha=0.7, s=60)
ax3.scatter(slopes[~cal_mask_vis], deltas[~cal_mask_vis], color='red', marker='s', label='Validation', alpha=0.7, s=60)

if np.std(slopes) > 1e-10:
    slope_fit, intercept_fit, _, _, _ = linregress(slopes, deltas)
    x_range = np.linspace(np.min(slopes), np.max(slopes), 100)
    ax3.plot(x_range, slope_fit * x_range + intercept_fit, 'gray', lw=1.5, alpha=0.5,
             label=f'Fit: y={slope_fit:.2f}x+{intercept_fit:.2f}')

ax3.axhline(0, color='black', lw=1)
ax3.axvline(0, color='black', lw=1, alpha=0.5)
ax3.set_xlabel("Local Slope R'(t)", fontsize=9)
ax3.set_ylabel('Δ (signed)', fontsize=9)
ax3.set_title("C) Δ vs Local Slope R'(t)\n(Cal=●, Val=■)", fontsize=10, fontweight='bold')
ax3.legend(fontsize=7)
ax3.grid(True, alpha=0.3)

# --- Panel D: Δ vs Tangent Angle ---
ax4 = fig.add_subplot(gs[1, 0])
tangents = np.array([r['geometry']['tangent_angle'] for r in all_results])

ax4.scatter(tangents[cal_mask_vis], deltas[cal_mask_vis], color='blue', label='Calibration', alpha=0.7, s=60)
ax4.scatter(tangents[~cal_mask_vis], deltas[~cal_mask_vis], color='red', marker='s', label='Validation', alpha=0.7, s=60)

if np.std(tangents) > 1e-10:
    tan_slope, tan_intercept, _, _, _ = linregress(tangents, deltas)
    x_range = np.linspace(np.min(tangents), np.max(tangents), 100)
    ax4.plot(x_range, tan_slope * x_range + tan_intercept, 'gray', lw=1.5, alpha=0.5,
             label=f'Fit: y={tan_slope:.2f}x+{tan_intercept:.2f}')

ax4.axhline(0, color='black', lw=1)
ax4.axvline(0, color='black', lw=1, alpha=0.5)
ax4.set_xlabel('Tangent Angle (rad)', fontsize=9)
ax4.set_ylabel('Δ (signed)', fontsize=9)
ax4.set_title('D) Δ vs Tangent Angle\n(strongest correlator)', fontsize=10, fontweight='bold')
ax4.legend(fontsize=7)
ax4.grid(True, alpha=0.3)

# --- Panel E: Correlation Heatmap ---
ax5 = fig.add_subplot(gs[1, 1])
features_for_corr = np.column_stack([slopes, tangents,
                                      np.array([r['geometry']['complex_phase_arg_DN'] for r in all_results]),
                                      deltas])
feature_names = ["Slope R'", 'Tan Angle', 'arg D_N', 'Δ']
corr_matrix = np.corrcoef(features_for_corr.T)
im = ax5.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(im, ax=ax5, label='Correlation')
ax5.set_xticks(range(4))
ax5.set_yticks(range(4))
ax5.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=8)
ax5.set_yticklabels(feature_names, fontsize=8)
ax5.set_title('E) Feature Correlations', fontsize=10, fontweight='bold')

# Annotate cells
for ii in range(4):
    for jj in range(4):
        ax5.text(jj, ii, f'{corr_matrix[ii, jj]:.2f}', ha='center', va='center', fontsize=7,
                 color='white' if abs(corr_matrix[ii, jj]) > 0.5 else 'black')

# --- Panel F: Error Distribution ---
ax6 = fig.add_subplot(gs[1, 2])
errors_before = deltas.copy()

# Simple linear correction for comparison
if np.std(slopes) > 1e-10:
    slope_corr, int_corr, _, _, _ = linregress(slopes, deltas)
    errors_after = deltas - (slope_corr * slopes + int_corr)
else:
    errors_after = errors_before.copy()

ax6.hist(errors_before, bins=12, color='red', alpha=0.5, label='Before (raw)', edgecolor='k')
ax6.hist(errors_after, bins=12, color='green', alpha=0.5, label='After (linear correction)', edgecolor='k')
ax6.axvline(0, color='black', lw=1.5)
ax6.set_xlabel('Residual Error', fontsize=9)
ax6.set_ylabel('Count', fontsize=9)
ax6.set_title('F) Raw vs Linearly-Corrected Errors\n(Not final method!)', fontsize=10, fontweight='bold')
ax6.legend(fontsize=7)
ax6.grid(True, alpha=0.3, axis='y')

# --- Panel G: Two-Stage Process ---
ax7 = fig.add_subplot(gs[1, 3])
ax7.axis('off')
ax7.set_title("G) Cipher's Two-Stage Workflow", fontsize=10, fontweight='bold')

workflow_text = (
    "TWO-STAGE RESIDUAL ANALYSIS\n\n"
    "Step 1: Predict SIGN(Δ)\n"
    "  ├─ Features: R'(t), R''(t), arg D_N,\n"
    "  │            tangent_angle, hop_frac\n"
    "  └─ Check: Sign accuracy > 50%?\n\n"
    "Step 2: Calibrate MAGNITUDE |Δ|\n"
    "  ├─ Only for correct-sign cases\n"
    "  └─ Check: |Δ| predictable?\n\n"
    "Result: SIGN_pred × |Δ|_calibrated\n"
    "       → Final Δ_pred\n\n"
    "⚠ NO joint regression until\n"
    "  both stages succeed!"
)
ax7.text(0.05, 0.95, workflow_text, transform=ax7.transAxes,
         fontsize=7.5, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='#f8f9fa', alpha=0.8))

# --- Panel H: Cal vs Val Performance ---
ax8 = fig.add_subplot(gs[2, 0:2])

# Compute combined performance
cal_deltas_raw = deltas[cal_mask]
val_deltas_raw = deltas[~cal_mask]

# Corrected: apply sign × magnitude where sign correct
deltas_corrected = deltas.copy().astype(float)
for i, r in enumerate(all_results):
    if correct_sign_mask[i]:
        if cal_mask[i]:
            cal_idx = np.where(cal_mag_mask)[0]
            pos = np.where(cal_indices == i)[0]
            if len(pos) > 0:
                pred_mag = y_pred_cal_mag[pos[0]]
                pred_sign = sign_pred_cal[i]
                deltas_corrected[i] = pred_sign * pred_mag
        else:
            val_idx_local = i - val_offset
            val_positions = np.where(val_indices == i)[0]
            if len(val_positions) > 0 and val_positions[0] < len(y_pred_val_mag):
                pred_mag = y_pred_val_mag[val_positions[0]]
                pred_sign = sign_pred_val[val_idx_local]
                deltas_corrected[i] = pred_sign * pred_mag

cal_deltas_corr = deltas_corrected[cal_mask]
val_deltas_corr = deltas_corrected[~cal_mask]

categories = ['Calibration\n(1-10)', 'Validation\n(11-20)']
before_means = [np.mean(np.abs(cal_deltas_raw)), np.mean(np.abs(val_deltas_raw))]
after_means = [np.mean(np.abs(cal_deltas_corr)), np.mean(np.abs(val_deltas_corr))]

x_pos = np.arange(len(categories))
width = 0.35
bars1 = ax8.bar(x_pos - width/2, before_means, width, label='Before correction', color='#e74c3c', alpha=0.7)
bars2 = ax8.bar(x_pos + width/2, after_means, width, label='After correction', color='#2ecc71', alpha=0.7)

ax8.set_ylabel('Mean |Δ|', fontsize=9)
ax8.set_title('H) Calibration vs Validation Performance\n(Two-Stage Method)', fontsize=10, fontweight='bold')
ax8.set_xticks(x_pos)
ax8.set_xticklabels(categories, fontsize=8)
ax8.legend(fontsize=7)
ax8.grid(True, alpha=0.3, axis='y')

for bar in bars1:
    ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=7)
for bar in bars2:
    ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=7)

# --- Panel I: Coverage Profile ---
ax9 = fig.add_subplot(gs[2, 2])
tolerances = [2.0, 1.5, 1.0, 0.5, 0.25]
covers_before = [np.mean(np.abs(deltas) < d) for d in tolerances]
covers_after = [np.mean(np.abs(deltas_corrected) < d) for d in tolerances]

ax9.step(tolerances, covers_before, 'r-o', where='mid', lw=2, label='Before correction', markersize=5)
ax9.step(tolerances, covers_after, 'g-o', where='mid', lw=2, label='After correction', markersize=5)
ax9.set_xlabel('Tolerance δ', fontsize=9)
ax9.set_ylabel('Coverage C₂₀(δ)', fontsize=9)
ax9.set_title('I) Coverage Profile', fontsize=10, fontweight='bold')
ax9.legend(fontsize=7)
ax9.grid(True, alpha=0.3)
ax9.set_xticks(tolerances)
ax9.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

# --- Panel J: Summary Table ---
ax10 = fig.add_subplot(gs[2, 3])
ax10.axis('off')
ax10.set_title('J) Summary', fontsize=10, fontweight='bold')

strongest_key = max(correlations, key=lambda k: abs(correlations[k]))
strongest_val = correlations[strongest_key]

summary_lines = [
    f"Original mean|Δ|:      {np.mean(np.abs(deltas)):.3f}",
    f"Corrected mean|Δ|:    {np.mean(np.abs(deltas_corrected)):.3f}",
    "",
    f"Sign acc (cal):  {acc_cal*100:.0f}%",
    f"Sign acc (val):  {acc_val*100:.0f}%",
    "",
    f"Mag MAE (cal):   {mae_cal:.3f}",
    f"Mag MAE (val):   {mae_val:.3f}" if not np.isnan(mae_val) else "Mag MAE (val):   N/A",
    "",
    f"Strongest: {strongest_key[:12]}",
    f"  r = {strongest_val:+.3f}",
]

summary_text = "\n".join(summary_lines)
ax10.text(0.05, 0.95, summary_text, transform=ax10.transAxes,
          fontsize=8, verticalalignment='top', fontfamily='monospace',
          bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# --- Panel K: Full Residual Comparison Table ---
ax_table = fig.add_subplot(gs[3, :])
ax_table.axis('off')
ax_table.set_title('K) COMPLETE TWO-STAGE RESIDUAL TABLE', fontsize=11, fontweight='bold')

table_data = []
for i, r in enumerate(all_results):
    set_label = 'CAL' if r['is_calibration'] else 'VAL'
    sign_ok = "✓" if correct_sign_mask[i] else "✗"
    actual_delta = r['delta_actual']
    corrected_delta = deltas_corrected[i]
    improvement = abs(actual_delta) - abs(corrected_delta)
    impr_str = f"{improvement:+.3f}" if improvement != 0 else "  0.000"
    table_data.append([
        f"#{r['zero_index']}",
        set_label,
        f"{r['t_known']:.4f}",
        f"{r['delta_actual']:+.4f}",
        sign_ok,
        f"{corrected_delta:+.4f}",
        f"{abs(corrected_delta):.4f}",
        impr_str
    ])

# Summary row
table_data.append([
    'SUMMARY', '',
    f'Cal mean|Δ|: {np.mean(np.abs(cal_deltas_corr)):.3f}',
    f'Val mean|Δ|: {np.mean(np.abs(val_deltas_corr)):.3f}',
    f'{int(np.sum(correct_sign_mask))}/20',
    f'Overall: {np.mean(np.abs(deltas_corrected)):.3f}',
    '',
    f'Impr: {(np.mean(np.abs(deltas))-np.mean(np.abs(deltas_corrected)))/np.mean(np.abs(deltas))*100:.0f}%'
])

table = ax_table.table(cellText=table_data,
                       colLabels=['#', 'Set', 't_known', 'Δ_actual', 'Sign', 'Δ_corrected', '|Δ_corr|', 'Impr.'],
                       loc='center', cellLoc='center',
                       colWidths=[0.06, 0.04, 0.08, 0.10, 0.06, 0.10, 0.08, 0.08])
table.auto_set_font_size(False)
table.set_fontsize(7)
table.scale(1.0, 1.4)

for j in range(8):
    table[0, j].set_facecolor('#6d4aff')
    table[0, j].set_text_props(color='white', weight='bold')

for i in range(1, len(table_data)):
    for j in range(8):
        if i % 2 == 0:
            table[i, j].set_facecolor('#f0f0f0')
        # Color sign column
        if j == 4:
            if table_data[i-1][4] == '✓':
                table[i, j].set_facecolor('#d5f5e3')
            else:
                table[i, j].set_facecolor('#fadbd8')
    # Summary row
    if i == len(table_data):
        for j in range(8):
            table[i, j].set_facecolor('#f9e79f')
            table[i, j].set_text_props(weight='bold')

# Super-title
fig.suptitle(f'SESSION 123: GEOMETRIC NORMALIZATION & TWO-STAGE RESIDUAL ANALYSIS\n'
             f"Cipher's Protocol: Separate SIGN prediction from MAGNITUDE calibration\n"
             f'α={ALPHA}, β={BETA}, γ={GAMMA}, λ={LAM_OPT}, φ={PHI_OPT}',
             fontsize=13, fontweight='bold', y=0.995)

plt.savefig('session123_geometric_normalization.png', dpi=200, bbox_inches='tight')
print("\n  Plot gespeichert: session123_geometric_normalization.png")

# ============================================================
# FAZIT
# ============================================================

print("\n" + "=" * 80)
print("FAZIT — SESSION 123: TWO-STAGE ANALYSIS RESULTS")
print("=" * 80)

# --- FIX: Define next_step_msg ---
if acc_val > 0.60:
    next_step_msg = "✓ Sign prediction has signal — refine magnitude calibration"
elif acc_val > 0.50:
    next_step_msg = "⚠ Marginal sign prediction — try nonlinear classifier"
else:
    next_step_msg = "✗ Sign prediction at chance — reconsider feature set"

print(f"""
  CIPHER'S HYPOTHESIS TESTED:
  ─────────────────────────────────────────
  • Original mean|Δ| = {np.mean(np.abs(deltas)):.4f}
  • Corrected mean|Δ| = {np.mean(np.abs(deltas_corrected)):.4f}
  • Sign prediction accuracy (cal): {acc_cal*100:.1f}%
  • Sign prediction accuracy (val): {acc_val*100:.1f}%
  • Magnitude MAE (cal): {mae_cal:.4f}
  • Magnitude MAE (val): {mae_val:.4f}

  INTERPRETATION:
  ─────────────────────────────────────────
  If sign accuracy >> 50%, then Δ has GEOMETRIC PREDICTORS.
  If sign accuracy ≈ 50%, then errors may be RANDOM.

  STRONGEST GEOMETRIC CORRELATOR:
  ─────────────────────────────────────────
  {strongest_key}: r = {strongest_val:+.4f}

  NEXT ACTION ITEMS:
  ─────────────────────────────────────────
  1. ✓ Store full geometry (done)
  2. ✓ Test sign prediction separately (Phase 1 complete)
  3. {next_step_msg}
  4. Only THEN consider joint regression

  KEY INSIGHT:
  ─────────────────────────────────────────
  The +3 to +4.5 errors ARE systematic (not random).
  The question is: Can geometry-based normalization fix them?
""")
print("=" * 80)