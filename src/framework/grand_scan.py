#!/usr/bin/env python3
"""
session29_grand_scan_with_zeros.py
Session 29: Der Grand Unified Scan (10 bis 50000) MIT Nullstellen-Validierung.

FEATURES:
1. Lädt zeros1.txt für Live-Validierung.
2. Durchgehender Scan von t=10 bis t=50000.
3. Adaptive Schrittweite.
4. Fortsetzungsschutz.
5. Live-Statistik: Wie viele Punkte liegen auf Nullstellen?
"""

import numpy as np
import pandas as pd
from pathlib import Path
import time
import sys
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# KONFIGURATION
# ============================================================

OUTPUT_FILE = Path("grand_scan_10_50000.csv")
ZEROS_FILE = Path("zeros1.txt")
N_TERMS = 5000

T_START = 10.0
T_END = 50000.0

# Adaptive Schrittweiten
SCENARIOS = [
    (10.0, 1000.0, 0.5),
    (1000.0, 5000.0, 1.0),
    (5000.0, 15000.0, 2.0),
    (15000.0, 30000.0, 5.0),
    (30000.0, 50000.0, 10.0)
]

# Toleranz, um eine Nullstelle zu "treffen" (in t-Einheiten)
ZERO_TOLERANCE = 0.1

# ============================================================
# HILFSFUNKTIONEN
# ============================================================

def load_zeros(filepath):
    """Lädt alle Nullstellen in ein numpy Array."""
    if not filepath.exists():
        print(f"⚠️  Warnung: {filepath} nicht gefunden! Validierung deaktiviert.")
        return np.array([])
    lines = filepath.read_text(encoding='utf-8').strip().split('\n')
    zeros = np.array([float(l.strip()) for l in lines if l.strip() and not l.startswith('#')])
    print(f"✅ Geladene {len(zeros)} Nullstellen aus {filepath.name}")
    return zeros

def compute_RN_batch_fast(t_values, sigma_array, N=N_TERMS):
    ns = np.arange(1, N + 1, dtype=float)
    log_ns = np.log(ns)
    mags = ns[:, None, None] ** (-sigma_array[None, None, :])
    phases = np.exp(-1j * log_ns[:, None, None] * t_values[None, :, None])
    terms = mags * phases
    cum_sum = np.cumsum(terms, axis=0)
    abs_sum = np.abs(cum_sum)
    return np.min(abs_sum, axis=0)

def measure_radius_fast(gamma, N=N_TERMS):
    rho_pred = 17.2 / gamma
    coarse_max = max(0.01, min(1.5 * rho_pred, 0.3)) 
    coarse_step = max(0.0005, 0.15 * rho_pred)
    
    deltas = np.arange(0, coarse_max + coarse_step, coarse_step)
    t_pos = gamma + deltas
    t_neg = gamma - deltas
    
    sigma_low = np.linspace(0.0, 0.5, 50)
    sigma_high = np.linspace(0.5, 1.0, 50)
    
    rn_low_pos = compute_RN_batch_fast(t_pos, sigma_low, N)
    rn_low_neg = compute_RN_batch_fast(t_neg, sigma_low, N)
    rn_high_pos = compute_RN_batch_fast(t_pos, sigma_high, N)
    rn_high_neg = compute_RN_batch_fast(t_neg, sigma_high, N)
    
    min_low_pos = np.min(rn_low_pos, axis=1)
    min_low_neg = np.min(rn_low_neg, axis=1)
    min_high_pos = np.min(rn_high_pos, axis=1)
    min_high_neg = np.min(rn_high_neg, axis=1)
    
    labels_pos = (min_high_pos < min_low_pos).astype(float)
    labels_neg = (min_high_neg < min_low_neg).astype(float)
    rates = (labels_pos + labels_neg) / 2.0
    
    idx_cross = np.where(rates < 0.5)[0]
    
    if len(idx_cross) == 0:
        return coarse_max, "Upper"
    
    idx = idx_cross[0]
    if idx == 0:
        return 0.0005, "Lower"
    
    x1, y1 = deltas[idx-1], rates[idx-1]
    x2, y2 = deltas[idx], rates[idx]
    
    if y2 != y1:
        radius = x1 + (0.5 - y1) * (x2 - x1) / (y2 - y1)
    else:
        radius = x1
    
    ratio = radius / rho_pred if rho_pred > 0 else 0
    if ratio < 0.1:
        status = "Lower"
    elif ratio > 3.0:
        status = "Upper"
    else:
        status = "Found"
    
    return radius, status

def save_progress(df, filepath):
    df.to_csv(filepath, index=False)

def check_zero_hit(t_val, zeros, tolerance=ZERO_TOLERANCE):
    """Prüft, ob t_val nahe einer Nullstelle liegt."""
    if len(zeros) == 0:
        return False
    # Schnelle Suche: Finde den nächsten Eintrag
    idx = np.searchsorted(zeros, t_val)
    # Prüfe links und rechts
    if idx > 0 and abs(zeros[idx-1] - t_val) < tolerance:
        return True
    if idx < len(zeros) and abs(zeros[idx] - t_val) < tolerance:
        return True
    return False

# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("=" * 70)
    print("SESSION 29: GRAND UNIFIED SCAN (MIT NULLSTELLEN)")
    print("=" * 70)
    
    # 1. Nullstellen laden
    zeros = load_zeros(ZEROS_FILE)
    
    start_time = time.time()
    
    # Prüfen auf Fortsetzung
    start_t = T_START
    existing_df = pd.DataFrame()
    
    if OUTPUT_FILE.exists():
        print("⚠️  Datei existiert. Fortsetzung wird geprüft...")
        try:
            existing_df = pd.read_csv(OUTPUT_FILE)
            if len(existing_df) > 0:
                last_t = existing_df['t'].iloc[-1]
                if last_t >= T_END:
                    print("✅ Scan bereits abgeschlossen!")
                    exit(0)
                start_t = last_t + 0.1
                print(f"Fortsetzung ab t = {start_t:.1f}")
        except Exception as e:
            print(f"⚠️  Fehler beim Laden: {e}. Neustart.")
            existing_df = pd.DataFrame()
    
    results = []
    hits_on_zeros = 0
    total_scanned = 0
    
    # 2. Scan Loop
    for t_min, t_max, step in SCENARIOS:
        if t_min >= T_END:
            break
        
        current_start = max(t_min, start_t)
        if current_start >= t_max:
            continue
            
        print(f"\n>>> Bereich: {t_min:.0f} - {t_max:.0f} (Schritt {step})")
        
        t_current = current_start
        count_in_range = 0
        
        while t_current < t_max and t_current < T_END:
            try:
                r, s = measure_radius_fast(t_current)
                
                # Live-Validierung
                is_hit = check_zero_hit(t_current, zeros)
                if is_hit:
                    hits_on_zeros += 1
                
                results.append({'t': t_current, 'rho': r, 'status': s})
                total_scanned += 1
                count_in_range += 1
                
                # Fortschritt
                if count_in_range % 100 == 0:
                    elapsed = time.time() - start_time
                    eta = (elapsed / total_scanned) * ((T_END - t_current) / step)
                    hit_rate = (hits_on_zeros / total_scanned * 100) if total_scanned > 0 else 0
                    print(f"  t={t_current:.1f} | {s:6s} | Hits={hits_on_zeros} ({hit_rate:.1f}%) | ETA: {eta/3600:.1f}h")
                
                # Speichern alle 500 Punkte
                if count_in_range % 500 == 0:
                    temp_df = pd.DataFrame(results)
                    if len(existing_df) > 0:
                        combined = pd.concat([existing_df, temp_df], ignore_index=True)
                    else:
                        combined = temp_df
                    save_progress(combined, OUTPUT_FILE)
                
                t_current += step
                
            except Exception as e:
                print(f"  ❌ Fehler bei t={t_current}: {e}")
                t_current += step
        
        # Nach Bereich speichern
        temp_df = pd.DataFrame(results)
        if len(existing_df) > 0:
            combined = pd.concat([existing_df, temp_df], ignore_index=True)
        else:
            combined = temp_df
        save_progress(combined, OUTPUT_FILE)
        print(f"  ✅ Bereich abgeschlossen. {count_in_range} Punkte. Hits: {hits_on_zeros}")

    # Finale Speicherung
    final_df = pd.DataFrame(results)
    if len(existing_df) > 0:
        final_df = pd.concat([existing_df, final_df], ignore_index=True)
    save_progress(final_df, OUTPUT_FILE)
    
    print("\n" + "=" * 70)
    print("SCAN ABGESCHLOSSEN!")
    print(f"Endzeit: {time.strftime('%H:%M:%S')}")
    print(f"Gesamte Punkte: {total_scanned}")
    print(f"Punkte auf Nullstellen: {hits_on_zeros} ({100*hits_on_zeros/total_scanned:.2f}%)")
    print(f"Ausgabedatei: {OUTPUT_FILE}")
    print("=" * 70)