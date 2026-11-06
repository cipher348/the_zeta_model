#!/usr/bin/env python3
"""
Session 132 Part A: Feature Generator (INTEGRATED WITH 05_geometric_normalization.py)
====================================================================================
Reads zeros1.txt → computes Duffing-Zeta features → saves to CSV/JSON

CORE: Uses your duffing_zeta_with_full_geometry() from 05_geometric_normalization.py
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime

# ============================================================
# IMPORT YOUR DUFFING-ZETA FUNCTION FROM 05_geometric_normalization.py
# ============================================================

# Copied directly from 05_geometric_normalization.py (lines 63-148):

ALPHA = 12.89
BETA  = 370.92
GAMMA = 0.4811
LAM_OPT = 112.4
PHI_OPT = 2.035

X_STAR = np.sqrt(ALPHA / BETA)

def duffing_zeta_with_full_geometry(s_complex, N=5000, h=0.01,
                                     lam=LAM_OPT, phi=PHI_OPT):
    """
    EXACT COPY from 05_geometric_normalization.py
    Computes Duffing response and extracts geometry at a given zero position.
    """
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
    win_start = start_idx + max(0, (N - start_idx) - 5000)
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


def detect_zero_position(zero_known, N_scan=200, scan_window=2.0):
    """
    Finds the detected zero position by scanning nearby values.
    Exact copy from 05_geometric_normalization.py main loop.
    
    Returns:
        t_detected: The position where |Duffing| is minimized
        delta_actual: t_detected - zero_known
    """
    t_local = np.linspace(max(0.1, zero_known - scan_window), 
                          zero_known + scan_window, N_scan)
    abs_vals = []

    for tv in t_local:
        s_val = 0.5 + 1j * tv
        zd, _ = duffing_zeta_with_full_geometry(s_val, N=5000)
        abs_vals.append(abs(zd))

    min_idx = np.argmin(abs_vals)
    t_detected = t_local[min_idx]
    delta_actual = t_detected - zero_known
    
    return t_detected, delta_actual


# ============================================================
# CONFIGURATION
# ============================================================

CONFIG = {
    'input_file': r'C:\Users\k5d-q\OneDrive\Python\Zeta_PR\RELEVANT\PLAN C\SESSION19\zeros1.txt',
    'output_csv': r'C:\Users\k5d-q\OneDrive\Python\Zeta_PR\RELEVANT\PLAN C\SESSION19\zeta_features_5000.csv',
    'output_json': r'C:\Users\k5d-q\OneDrive\Python\Zeta_PR\RELEVANT\PLAN C\SESSION19\zeta_features_5000.json',
    'num_zeros': 5000,  # Number of zeros to extract
    'first_zero_offset': 1,  # Which zero to start from (1-indexed)
    'split_ratio': 0.5,  # 50% CAL (first half), 50% VAL (second half)
}

print("=" * 70)
print("SESSION 132 PART A: FEATURE GENERATOR")
print("Integrated with 05_geometric_normalization.py")
print("=" * 70)
print(f"\n  Input file:    {Path(CONFIG['input_file']).name}")
print(f"  Num zeros:     {CONFIG['num_zeros']}")
print(f"  Output CSV:    {Path(CONFIG['output_csv']).name}")
print(f"  Output JSON:   {Path(CONFIG['output_json']).name}")
print(f"  Split:         {int(CONFIG['split_ratio']*100)}% CAL / {int((1-CONFIG['split_ratio'])*100)}% VAL")
print(f"  Duffing params: α={ALPHA}, β={BETA}, γ={GAMMA}, λ={LAM_OPT}, φ={PHI_OPT}")


# ============================================================
# STEP 1: LOAD ZETA ZERO POSITIONS
# ============================================================

def load_zeta_zeros(input_path: str, num_zeros: int, start_from: int = 1) -> np.ndarray:
    print(f"\n[1/5] Loading {num_zeros} zeros from {input_path}...")
    
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"    Found {len(lines)} total lines in file")
    
    if len(lines) < num_zeros + start_from - 1:
        raise ValueError(f"File has only {len(lines)} zeros, but {num_zeros} requested")
    
    zeros = []
    for i in range(start_from - 1, start_from - 1 + num_zeros):
        line = lines[i].strip()
        if '\t' in line:
            parts = line.split('\t')
            value = float(parts[-1])
        else:
            value = float(line)
        zeros.append(value)
    
    zeros_array = np.array(zeros)
    print(f"    ✓ Loaded {len(zeros_array)} zeros:")
    print(f"      Range: [{zeros_array.min():.6f}, {zeros_array.max():.6f}]")
    print(f"      First 5: {zeros_array[:5]}")
    
    return zeros_array

# ===================================================================
# MISSING CALL: Actually invoke the function to populate 'zeros'
# ===================================================================

zeros = load_zeta_zeros(
    input_path=CONFIG['input_file'],
    num_zeros=CONFIG['num_zeros'],
    start_from=CONFIG['first_zero_offset']
)

# ============================================================
# STEP 2: PROCESS EACH ZERO WITH DUDDING-ZETA
# ============================================================

print(f"\n[2/5] Processing {CONFIG['num_zeros']} zeros through Duffing-Zeta...")
print("      This may take several minutes...")

results = []
total_time_start = datetime.now()

for idx in range(CONFIG['num_zeros']):
    zero_known = zeros[idx]
    zero_idx_1based = idx + 1
    set_label = 'CAL' if zero_idx_1based <= CONFIG['num_zeros'] * CONFIG['split_ratio'] else 'VAL'
    
    # Detect zero position
    t_detected, delta_actual = detect_zero_position(zero_known)
    
    # Compute features at detected position
    s_at_min = 0.5 + 1j * t_detected
    _, geometry = duffing_zeta_with_full_geometry(s_at_min, N=5000)
    
    # Store results
    result = {
        'idx': zero_idx_1based,
        'set': set_label,
        't_known': zero_known,
        't_detected': t_detected,
        'delta_actual': delta_actual,
        'R_prime': geometry['slope_R_prime'],
        'R_dbl': geometry['curvature_R_double_prime'],
        'arg_DN': geometry['complex_phase_arg_DN'],
        'tan_angle': geometry['tangent_angle'],
        'hopping_frac': geometry['hopping_frac'],
        'sync_index': geometry['sync_index'],
        'response_magnitude': geometry['response_magnitude'],
        'x_re_mean': geometry['x_re_mean'],
        'x_im_mean': geometry['x_im_mean'],
        'v_re_std': geometry['v_re_std'],
        'v_im_std': geometry['v_im_std'],
    }
    results.append(result)
    
    # Progress report every 250 zeros
    if (idx + 1) % 250 == 0:
        elapsed = (datetime.now() - total_time_start).total_seconds()
        eta = (CONFIG['num_zeros'] - (idx + 1)) * elapsed / (idx + 1)
        print(f"    Processed {idx + 1}/{CONFIG['num_zeros']} zeros... ETA: {eta:.0f}s")
        print(f"      Last: t={zero_known:.4f}, Δ={delta_actual:+.4f}, R'={geometry['slope_R_prime']:.4f}")

print(f"    ✓ Completed {len(results)} zeros in {(datetime.now()-total_time_start).total_seconds():.1f}s")


# ============================================================
# STEP 3: CREATE PRE-TARGET FEATURES (FOR GAIN ANALYSIS)
# ============================================================

print(f"\n[3/5] Computing pre-target features for gain analysis...")

df = pd.DataFrame(results)

# Cipher's pre-target features (NO delta_actual!)
df['phi_diff'] = (df['arg_DN'] - df['tan_angle'] + np.pi) % (2 * np.pi) - np.pi
df['phi_diff_deg'] = np.degrees(df['phi_diff'])
df['tan_dominance'] = np.abs(np.sin(df['phi_diff']))
df['norm_dominance'] = np.abs(np.cos(df['phi_diff']))
df['R_prime_abs'] = df['R_prime'].abs()
df['R_dbl_abs'] = df['R_dbl'].abs()
df['total_weighted'] = (df['R_prime_abs'] * df['tan_dominance'] + 
                        df['R_dbl_abs'] * df['norm_dominance'])

# Additional transforms
df['sin_tan_angle'] = np.sin(df['tan_angle'])
df['cos_tan_angle'] = np.cos(df['tan_angle'])
df['sin_arg_DN'] = np.sin(df['arg_DN'])
df['cos_arg_DN'] = np.cos(df['arg_DN'])
df['sin_phi_diff'] = np.sin(df['phi_diff'])
df['cos_phi_diff'] = np.cos(df['phi_diff'])
df['R_prime_sq'] = df['R_prime'] ** 2
df['R_dbl_sq'] = df['R_dbl'] ** 2

print(f"    ✓ Added {len([c for c in df.columns if c not in results[0].keys()])} derived features")


# ============================================================
# STEP 4: COMPUTE GAIN METRIC (DIAGNOSTIC TARGET)
# ============================================================

print(f"\n[4/5] Computing observed gain g_obs = Δ / total_weighted...")

df['g_obs'] = df['delta_actual'] / (df['total_weighted'] + 1e-8)
df['g_obs_abs'] = df['g_obs'].abs()
df['log_g_obs'] = np.sign(df['g_obs']) * np.log(df['g_obs_abs'] + 1e-8)

print(f"    ✓ Gain distribution:")
print(f"      Mean: {df['g_obs'].mean():+.6f}")
print(f"      Median: {df['g_obs'].median():+.6f}")
print(f"      Std: {df['g_obs'].std():.6f}")
print(f"      Range: [{df['g_obs'].min():+.6f}, {df['g_obs'].max():+.6f}]")


# ============================================================
# STEP 5: SAVE OUTPUTS
# ============================================================

print(f"\n[5/5] Saving outputs...")

# --- CSV Format (compact feature set) ---
csv_cols = ['idx', 'set', 't_known', 't_detected', 'delta_actual', 
            'R_prime', 'R_dbl', 'arg_DN', 'tan_angle',
            'phi_diff', 'phi_diff_deg', 'tan_dominance', 'norm_dominance',
            'R_prime_abs', 'R_dbl_abs', 'total_weighted',
            'g_obs', 'g_obs_abs', 'log_g_obs',
            'hopping_frac', 'sync_index', 'response_magnitude']

df[csv_cols].to_csv(CONFIG['output_csv'], index=False)
print(f"    ✓ Saved {CONFIG['output_csv']} ({len(df)} rows, {len(csv_cols)} columns)")

# --- JSON Format (full data) ---
with open(CONFIG['output_json'], 'w', encoding='utf-8') as f:
    json.dump(df.to_dict('records'), f, indent=2)
print(f"    ✓ Saved {CONFIG['output_json']}")

# --- Quick validation sample ---
print(f"\n    Sample (first 5 rows):")
print(df.head(5)[['idx', 'set', 'delta_actual', 'R_prime', 'R_dbl', 'g_obs']].to_string())


# ============================================================
# READY FOR SESSION 132 ANALYZER
# ============================================================

print(f"\n{'=' * 70}")
print("FEATURE GENERATION COMPLETE")
print(f"{'=' * 70}")
print(f"\n  Next step: Run session132_analyzer.py on the generated files")
print(f"  Expected runtime for 5000 zeros: ~5-10 hours")
print(f"\n  Generated files:")
print(f"    • {Path(CONFIG['output_csv']).resolve()}")
print(f"    • {Path(CONFIG['output_json']).resolve()}")
print(f"\n{'=' * 70}")