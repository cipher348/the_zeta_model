#!/usr/bin/env python3
"""
FEATURE GENERATOR v6.0 — SPECTRAL DYNAMICS
==========================================
Design principles:
  1. Compute ALL features at s = 0.5 + 1j*t_known (BEFORE detection)
  2. Multi-scale: extract from 3 trajectory segments (early/mid/late)
  3. Spectral: FFT-based features + resonance proximity
  4. Dynamic: energy, stability, convergence rate
  5. Zero leakage: delta_actual and t_detected never touch this pipeline

Output: zeta_features_v6_10000.csv
Expected runtime: ~14-18 hours (similar to v5)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# DUFFING PARAMETERS (frozen)
# ============================================================
ALPHA = 12.89
BETA  = 370.92
GAMMA = 0.4811
LAM_OPT = 112.4
PHI_OPT = 2.035
X_STAR = np.sqrt(ALPHA / BETA)

AVG_WINDOW = 2000
N_FULL = 5000
h = 0.01


# ============================================================
# CORE: FULL TRAJECTORY (returns raw arrays)
# ============================================================
def compute_full_trajectory(sigma, t_val, N=N_FULL, h=h, lam=LAM_OPT, phi=PHI_OPT):
    """
    Computes the full Duffing trajectory for real and imaginary parts.
    Returns raw arrays for downstream feature extraction.
    """
    x_re, v_re = 0.0, 0.0
    x_im, v_im = 0.0, 0.0

    traj_x_re = np.zeros(N)
    traj_v_re = np.zeros(N)
    traj_x_im = np.zeros(N)
    traj_v_im = np.zeros(N)
    traj_force_re = np.zeros(N)
    traj_force_im = np.zeros(N)

    for n in range(N):
        log_n = np.log(n + 1)
        mag = np.exp(-sigma * log_n)

        f_re = mag * np.cos(t_val * log_n + phi)
        f_im = -mag * np.sin(t_val * log_n + phi)

        # Real part trajectory
        a_re = ALPHA * x_re - BETA * x_re**3 - GAMMA * v_re + lam * f_re
        v_re += h * a_re
        x_re += h * v_re
        traj_x_re[n] = x_re
        traj_v_re[n] = v_re
        traj_force_re[n] = f_re

        # Imaginary part trajectory
        a_im = ALPHA * x_im - BETA * x_im**3 - GAMMA * v_im + lam * f_im
        v_im += h * a_im
        x_im += h * v_im
        traj_x_im[n] = x_im
        traj_v_im[n] = v_im
        traj_force_im[n] = f_im

    return {
        'x_re': traj_x_re, 'v_re': traj_v_re,
        'x_im': traj_x_im, 'v_im': traj_v_im,
        'force_re': traj_force_re, 'force_im': traj_force_im,
    }


# ============================================================
# FEATURE GROUP 1: MULTI-SCALE TEMPORAL
# ============================================================
def extract_multiscale_features(traj, N=N_FULL):
    """
    Extract features from 3 segments: early (0:1000), mid (2000:3000), late (3000:5000).
    Captures HOW the trajectory evolves over time.
    """
    segments = {
        'early': slice(0, min(1000, N)),
        'mid':   slice(2000, min(3000, N)),
        'late':  slice(max(0, N - AVG_WINDOW), N),
    }

    features = {}

    for seg_name, seg_slice in segments.items():
        x_re = traj['x_re'][seg_slice]
        x_im = traj['x_im'][seg_slice]
        v_re = traj['v_re'][seg_slice]
        v_im = traj['v_im'][seg_slice]

        # Position statistics
        features[f'{seg_name}_x_re_mean'] = float(np.mean(x_re))
        features[f'{seg_name}_x_im_mean'] = float(np.mean(x_im))
        features[f'{seg_name}_x_re_std'] = float(np.std(x_re))
        features[f'{seg_name}_x_im_std'] = float(np.std(x_im))

        # Velocity statistics
        features[f'{seg_name}_v_re_std'] = float(np.std(v_re))
        features[f'{seg_name}_v_im_std'] = float(np.std(v_im))

        # Magnitude (radius in complex plane)
        mag = np.sqrt(x_re**2 + x_im**2)
        features[f'{seg_name}_mag_mean'] = float(np.mean(mag))
        features[f'{seg_name}_mag_std'] = float(np.std(mag))
        features[f'{seg_name}_mag_max'] = float(np.max(mag))

        # Energy (kinetic + potential Duffing)
        # E = 0.5*v² + 0.5*alpha*x² + 0.25*beta*x⁴
        E_re = 0.5 * v_re**2 + 0.5 * ALPHA * x_re**2 + 0.25 * BETA * x_re**4
        features[f'{seg_name}_energy_mean'] = float(np.mean(E_re))
        features[f'{seg_name}_energy_std'] = float(np.std(E_re))

    # Convergence rate: how fast does magnitude stabilize?
    mag_early = features['early_mag_mean']
    mag_late = features['late_mag_mean']
    features['mag_convergence_ratio'] = float(mag_late / (mag_early + 1e-10))
    features['mag_change_rate'] = float((mag_late - mag_early) / (mag_early + 1e-10))

    # Velocity damping: how much does v decrease from early to late?
    v_early = features['early_v_re_std']
    v_late = features['late_v_re_std']
    features['vel_damping_ratio'] = float(v_late / (v_early + 1e-10))

    return features


# ============================================================
# FEATURE GROUP 2: SPECTRAL (FFT)
# ============================================================
def extract_spectral_features(traj, N=N_FULL, h=h):
    """
    FFT of the trajectory to capture frequency content.
    The driving force has frequency t_val — resonance effects show up here.
    """
    features = {}

    for component, label in [('x_re', 're'), ('x_im', 'im')]:
        signal = traj[component]

        # Apply window to reduce spectral leakage
        window = np.hanning(N)
        windowed = signal * window

        # FFT
        fft_vals = np.fft.rfft(windowed)
        freqs = np.fft.rfftfreq(N, d=h)
        power_spectrum = np.abs(fft_vals)**2

        # Top 3 dominant frequencies
        if len(power_spectrum) > 3:
            top_indices = np.argsort(power_spectrum)[::-1][:3]
            for k, idx in enumerate(top_indices):
                features[f'{label}_freq{k+1}_hz'] = float(freqs[idx])
                features[f'{label}_freq{k+1}_power'] = float(power_spectrum[idx])

        # Spectral centroid (weighted average frequency)
        total_power = np.sum(power_spectrum)
        if total_power > 1e-20:
            features[f'{label}_spectral_centroid'] = float(np.sum(freqs * power_spectrum) / total_power)
            features[f'{label}_spectral_spread'] = float(np.sqrt(np.sum((freqs - features[f'{label}_spectral_centroid'])**2 * power_spectrum) / total_power))
        else:
            features[f'{label}_spectral_centroid'] = 0.0
            features[f'{label}_spectral_spread'] = 0.0

        # Spectral entropy (how spread is the energy?)
        ps_norm = power_spectrum / (total_power + 1e-20)
        ps_norm = ps_norm[ps_norm > 0]
        features[f'{label}_spectral_entropy'] = float(-np.sum(ps_norm * np.log2(ps_norm))) if len(ps_norm) > 0 else 0.0

        # Power ratio: low freq vs high freq
        mid_freq = len(freqs) // 2
        low_power = np.sum(power_spectrum[:mid_freq])
        high_power = np.sum(power_spectrum[mid_freq:])
        features[f'{label}_low_high_ratio'] = float(low_power / (high_power + 1e-10))

    # Cross-spectral: coherence between real and imaginary
    fft_re = np.fft.rfft(traj['x_re'] * window)
    fft_im = np.fft.rfft(traj['x_im'] * window)
    cross_power = np.abs(fft_re * np.conj(fft_im))
    features['cross_spectral_max'] = float(np.max(cross_power)) if len(cross_power) > 0 else 0.0

    return features


# ============================================================
# FEATURE GROUP 3: RESONANCE PROXIMITY
# ============================================================
def extract_resonance_features(traj, t_val, N=N_FULL, h=h):
    """
    Measures how close the system is to resonance.
    The driving force oscillates at frequency t_val in log-space.
    The Duffing oscillator has natural frequency ~ sqrt(alpha).
    Resonance proximity should affect detection accuracy.
    """
    features = {}

    # Natural frequency of Duffing (small amplitude approximation)
    omega_natural = np.sqrt(ALPHA)
    features['omega_natural'] = float(omega_natural)

    # Driving frequency in real time (t_val * (1/n) → effective freq varies)
    # At the averaging window, the effective angular frequency is t_val / log(N)
    effective_freq = t_val / np.log(N)
    features['effective_driving_freq'] = float(effective_freq)

    # Resonance ratio
    features['resonance_ratio'] = float(effective_freq / (omega_natural + 1e-10))

    # Detuning: how far from resonance
    features['detuning'] = float(abs(effective_freq - omega_natural))

    # Amplitude growth rate: how fast does |x| grow initially?
    mag_seq = np.sqrt(traj['x_re']**2 + traj['x_im']**2)
    # Fit exponential growth rate to first 500 steps
    early_mag = mag_seq[:500]
    early_log = np.log(early_mag + 1e-20)
    time_axis = np.arange(500) * h
    if len(time_axis) > 2:
        slope, _ = np.polyfit(time_axis, early_log, 1)
        features['growth_rate'] = float(slope)
    else:
        features['growth_rate'] = 0.0

    # Amplitude at steady state (last 500 steps)
    late_mag = mag_seq[-500:]
    features['steady_amplitude'] = float(np.mean(late_mag))
    features['steady_amplitude_std'] = float(np.std(late_mag))

    # Oscillation count: how many zero crossings in the trajectory?
    sign_changes_re = np.sum(np.diff(np.sign(traj['x_re'][-2000:]) != 0))
    features['oscillation_count_re'] = int(sign_changes_re)
    sign_changes_im = np.sum(np.diff(np.sign(traj['x_im'][-2000:]) != 0))
    features['oscillation_count_im'] = int(sign_changes_im)

    # Phase drift: how much does the phase rotate over the averaging window?
    phase_seq = np.arctan2(traj['x_im'][-2000:], traj['x_re'][-2000:])
    phase_unwrapped = np.unwrap(phase_seq)
    features['phase_drift_rate'] = float(np.polyfit(np.arange(2000) * h, phase_unwrapped, 1)[0])

    return features


# ============================================================
# FEATURE GROUP 4: STABILITY & LYAPUNOV-LIKE
# ============================================================
def extract_stability_features(traj, N=N_FULL, h=h):
    """
    Approximate stability measures without full Lyapunov computation.
    Uses trajectory sensitivity indicators.
    """
    features = {}

    # Variance ratio: late vs early (should decrease if system stabilizes)
    var_early_re = np.var(traj['x_re'][:1000])
    var_late_re = np.var(traj['x_re'][-1000:])
    features['variance_ratio_re'] = float(var_late_re / (var_early_re + 1e-10))

    var_early_im = np.var(traj['x_im'][:1000])
    var_late_im = np.var(traj['x_im'][-1000:])
    features['variance_ratio_im'] = float(var_late_im / (var_early_im + 1e-10))

    # Autocorrelation decay: how fast does x[n] decorrelate from x[n-k]?
    x = traj['x_re'][-2000:]
    x = x - np.mean(x)
    if np.std(x) > 1e-10:
        acf_1 = np.correlate(x, x, mode='full')[len(x)-1]  # lag 0
        acf_10 = np.correlate(x[:-10], x[10:], mode='valid')[0] if len(x) > 10 else 0
        features['autocorr_decay_10'] = float(acf_10 / (acf_1 + 1e-10))
    else:
        features['autocorr_decay_10'] = 0.0

    # Kurtosis of position (heavy tails = unstable)
    from scipy.stats import kurtosis
    features['kurtosis_re'] = float(kurtosis(traj['x_re'][-2000:]))
    features['kurtosis_im'] = float(kurtosis(traj['x_im'][-2000:]))

    # Well occupancy: fraction of time in each Duffing well
    x_re = traj['x_re'][-2000:]
    features['well_positive_frac'] = float(np.mean(np.abs(x_re) > X_STAR * 0.8))
    features['well_transition_count'] = int(np.sum(np.abs(np.diff(np.sign(x_re - X_STAR * 0.5))) > 0))

    # Response curvature at the operating point
    mag = np.sqrt(traj['x_re'][-2000:]**2 + traj['x_im'][-2000:]**2)
    if len(mag) > 2:
        R_prime = np.gradient(mag, h)
        R_dbl = np.gradient(R_prime, h)
        features['slope_R_prime'] = float(np.mean(R_prime[-50:]))
        features['curvature_R_dbl'] = float(np.mean(R_dbl[-50:]))
        features['slope_var'] = float(np.var(R_prime[-200:]))
        features['curvature_var'] = float(np.var(R_dbl[-200:]))
    else:
        features['slope_R_prime'] = 0.0
        features['curvature_R_dbl'] = 0.0
        features['slope_var'] = 0.0
        features['curvature_var'] = 0.0

    return features


# ============================================================
# FEATURE GROUP 5: GEOMETRY (preserved from v5, verified clean)
# ============================================================
def extract_geometry_features(traj, avg_window=AVG_WINDOW, N=N_FULL):
    """
    Original geometric features from v5 — proven clean (no leakage).
    Computed at t_known, NOT at t_detected.
    """
    start_idx = N - avg_window
    win_x_re = traj['x_re'][start_idx:]
    win_x_im = traj['x_im'][start_idx:]
    win_v_re = traj['v_re'][start_idx:]
    win_v_im = traj['v_im'][start_idx:]

    avg_re = np.mean(win_x_re)
    avg_im = np.mean(win_x_im)
    zeta_d = avg_re + 1j * avg_im

    mag_seq = np.sqrt(win_x_re**2 + win_x_im**2)
    if len(mag_seq) > 2:
        from numpy import gradient
        R_prime = gradient(mag_seq, h)
        R_double_prime = gradient(R_prime, h)
        slope_at_min = float(np.mean(R_prime[-50:]))
        curvature_at_min = float(np.mean(R_double_prime[-50:]))
    else:
        slope_at_min = 0.0
        curvature_at_min = 0.0

    complex_phase = float(np.angle(zeta_d))

    dx_re = float(np.mean(np.diff(win_x_re)))
    dx_im = float(np.mean(np.diff(win_x_im)))
    tangent_angle = float(np.arctan2(dx_im, dx_re))

    phi_diff = (complex_phase - tangent_angle + np.pi) % (2 * np.pi) - np.pi

    log_n_array = np.log(np.arange(N) + 1)
    force_phase_tail = (zeta_d.imag * 0 + PHI_OPT)  # placeholder
    # Sync index
    force_sign = np.sign(np.cos(np.arange(start_idx, N) * 0 + PHI_OPT))
    vel_sign = np.sign(win_v_re[-50:])
    # Simplified: use actual force phase
    t_val_eff = float(np.angle(zeta_d))  # approximation
    force_phase = t_val_eff * log_n_array[start_idx:] + PHI_OPT
    force_sign = np.sign(np.cos(force_phase[-50:]))
    sync_index = float(np.mean(force_sign * vel_sign))

    features = {
        'R_prime': slope_at_min,
        'R_dbl': curvature_at_min,
        'arg_DN': complex_phase,
        'tan_angle': tangent_angle,
        'phi_diff': float(phi_diff),
        'hopping_frac': float(min(np.mean(np.abs(win_x_re) > X_STAR * 0.8),
                                  np.mean(np.abs(win_x_re) < X_STAR * 0.2))),
        'sync_index': sync_index,
        'response_magnitude': float(np.abs(zeta_d)),
        'x_re_mean': float(avg_re),
        'x_im_mean': float(avg_im),
        'v_re_std': float(np.std(win_v_re)),
        'v_im_std': float(np.std(win_v_im)),
        'total_weighted': float(abs(slope_at_min) * abs(np.sin(phi_diff)) +
                               abs(curvature_at_min) * abs(np.cos(phi_diff))),
    }

    return features


# ============================================================
# DETECTION (kept separate, AFTER all feature extraction)
# ============================================================
def detect_zero_position(zero_known, N_scan=200, scan_window=2.0,
                         N=N_FULL, h=h, lam=LAM_OPT, phi=PHI_OPT,
                         avg_window=AVG_WINDOW):
    """Same as v5 — scans nearby t-values to find minimum |Duffing|."""
    t_local = np.linspace(max(0.1, zero_known - scan_window),
                          zero_known + scan_window, N_scan)
    abs_vals = []

    for tv in t_local:
        s_val = 0.5 + 1j * tv
        traj = compute_full_trajectory(s_val.real, s_val.imag, N=N, h=h, lam=lam, phi=phi)
        avg_re = np.mean(traj['x_re'][N - avg_window:])
        avg_im = np.mean(traj['x_im'][N - avg_window:])
        zeta_d = avg_re + 1j * avg_im
        abs_vals.append(abs(zeta_d))

    min_idx = np.argmin(abs_vals)
    t_detected = t_local[min_idx]
    delta_actual = t_detected - zero_known

    return t_detected, delta_actual


# ============================================================
# MAIN PIPELINE
# ============================================================
CONFIG = {
    'input_file': r'C:\Users\k5d-q\OneDrive\Python\Zeta_PR\RELEVANT\PLAN C\SESSION19\zeros1.txt',
    'output_csv': r'C:\Users\k5d-q\OneDrive\Python\Zeta_PR\RELEVANT\beta_phase\session09\zeta_features_v6_10000.csv',
    'num_zeros': 10000,
    'first_zero_offset': 1,
    'split_ratio': 0.5,
}

print("=" * 70)
print("  FEATURE GENERATOR v6.0 — SPECTRAL DYNAMICS")
print("=" * 70)
print(f"\n  Design: Multi-scale + Spectral + Resonance + Stability + Geometry")
print(f"  Leakage: ZERO (all features computed at t_known before detection)")
print(f"  Expected features: ~60-70 per zero")
print(f"  Expected runtime: ~14-18 hours")

# Create output directory
Path(r'C:\Users\k5d-q\OneDrive\Python\Zeta_PR\RELEVANT\beta_phase\session09').mkdir(parents=True, exist_ok=True)

def load_zeta_zeros(input_path, num_zeros, start_from=1):
    print(f"\n[1/4] Loading {num_zeros} zeros...")
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    if len(lines) < start_from - 1 + num_zeros:
        raise ValueError(f"Not enough zeros in file: need {start_from - 1 + num_zeros}, have {len(lines)}")

    zeros = []
    for i in range(start_from - 1, start_from - 1 + num_zeros):
        line = lines[i].strip()
        if '\t' in line:
            value = float(line.split('\t')[-1])
        else:
            value = float(line)
        zeros.append(value)

    zeros_array = np.array(zeros)
    print(f"    Loaded {len(zeros_array)} zeros")
    print(f"      Range: [{zeros_array.min():.4f}, {zeros_array.max():.4f}]")
    return zeros_array

# ============================================================
# LOAD ZEROS
# ============================================================
zeros = load_zeta_zeros(
    input_path=CONFIG['input_file'],
    num_zeros=CONFIG['num_zeros'],
    start_from=CONFIG['first_zero_offset']
)

# ============================================================
# PROCESS ALL ZEROS
# ============================================================
print(f"\n[2/4] Processing {CONFIG['num_zeros']} zeros...")
print("      This will take approximately 14-18 hours...")

results = []
total_time_start = datetime.now()

for idx in range(CONFIG['num_zeros']):
    zero_known = zeros[idx]
    zero_idx_1based = idx + 1
    set_label = 'CAL' if zero_idx_1based <= CONFIG['num_zeros'] * CONFIG['split_ratio'] else 'VAL'

    # ============================================================
    # STEP 1: Compute trajectory at t_known (BEFORE detection)
    # ============================================================
    sigma = 0.5
    traj = compute_full_trajectory(sigma, zero_known)

    # ============================================================
    # STEP 2: Extract ALL features (no leakage possible)
    # ============================================================
    ms_feat = extract_multiscale_features(traj)
    spec_feat = extract_spectral_features(traj)
    res_feat = extract_resonance_features(traj, zero_known)
    stab_feat = extract_stability_features(traj)
    geo_feat = extract_geometry_features(traj)

    # ============================================================
    # STEP 3: Detection (separate, AFTER features)
    # ============================================================
    t_detected, delta_actual = detect_zero_position(zero_known)

    # ============================================================
    # STEP 4: Assemble result
    # ============================================================
    result = {
        'idx': zero_idx_1based,
        'set': set_label,
        't_known': float(zero_known),
        't_detected': float(t_detected),
        'delta_actual': float(delta_actual),
        'abs_delta': float(abs(delta_actual)),
        'log_t': float(np.log10(zero_known)),
    }

    # Merge all feature dicts
    for feat_dict in [ms_feat, spec_feat, res_feat, stab_feat, geo_feat]:
        for k, v in feat_dict.items():
            result[k] = float(v) if not isinstance(v, (int, np.integer)) else int(v)

    results.append(result)

    # Progress report every 500 zeros
    if (idx + 1) % 500 == 0:
        elapsed = (datetime.now() - total_time_start).total_seconds()
        eta = (CONFIG['num_zeros'] - (idx + 1)) * elapsed / (idx + 1)
        recent = results[-500:]
        recent_cal = [r['delta_actual'] for r in recent if r['set'] == 'CAL']
        recent_val = [r['delta_actual'] for r in recent if r['set'] == 'VAL']
        n_features = len([k for k in results[-1].keys() if k not in ['idx', 'set']])
        print(f"    [{datetime.now().strftime('%H:%M')}] Processed {idx + 1}/{CONFIG['num_zeros']}... ETA: {eta/3600:.1f}h")
        print(f"      Features per zero: {n_features}")
        if recent_cal:
            print(f"      Current CAL MAE: {np.mean(np.abs(recent_cal)):.4f}")
        if recent_val:
            print(f"      Current VAL MAE: {np.mean(np.abs(recent_val)):.4f}")

print(f"\n    Completed {len(results)} zeros in {(datetime.now()-total_time_start).total_seconds()/3600:.1f}h")

# ============================================================
# SAVE OUTPUT
# ============================================================
print(f"\n[3/4] Saving output...")
df = pd.DataFrame(results)

# Save CSV
csv_cols = [c for c in df.columns if c not in ['idx', 'set'] or c in ['idx', 'set']]
df.to_csv(CONFIG['output_csv'], index=False)
print(f"    Saved: {CONFIG['output_csv']}")
print(f"    Shape: {df.shape}")
print(f"    Features: {df.shape[1] - 5} (excluding idx, set, t_known, t_detected, delta_actual)")

# Summary JSON
summary = {
    'version': 'v6.0',
    'total_zeros': len(df),
    'n_features': df.shape[1] - 5,
    'feature_groups': {
        'multiscale': len(ms_feat),
        'spectral': len(spec_feat),
        'resonance': len(res_feat),
        'stability': len(stab_feat),
        'geometry': len(geo_feat),
    },
    'cal_mae': float(df[df['set']=='CAL']['abs_delta'].mean()),
    'val_mae': float(df[df['set']=='VAL']['abs_delta'].mean()),
    'runtime_hours': (datetime.now() - total_time_start).total_seconds() / 3600,
    'design_principles': [
        'All features computed at t_known BEFORE detection',
        'Multi-scale: early/mid/late trajectory segments',
        'Spectral: FFT frequency analysis',
        'Resonance: proximity to Duffing natural frequency',
        'Stability: variance ratios, autocorrelation, kurtosis',
        'Zero leakage: no feature uses t_detected or delta_actual',
    ],
}

import json
json_path = CONFIG['output_csv'].replace('.csv', '_summary.json')
with open(json_path, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"    Saved: {json_path}")

print(f"\n[4/4] Complete!")
print(f"\n{'='*70}")
print("  FEATURE GENERATOR v6.0 COMPLETE")
print(f"{'='*70}")
print(f"\n  Total features: {df.shape[1] - 5}")
print(f"  CAL MAE: {summary['cal_mae']:.4f}")
print(f"  VAL MAE: {summary['val_mae']:.4f}")
print(f"  Runtime: {summary['runtime_hours']:.1f}h")
print(f"\n  Feature breakdown:")
for group, count in summary['feature_groups'].items():
    print(f"    {group:15s}: {count} features")
print(f"\n  Output: {CONFIG['output_csv']}")