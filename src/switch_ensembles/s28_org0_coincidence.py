#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
s28_org0_coincidence.py — S28: ORG‑0 Koinzidenz‑Analyse (Grid vs. Produktion)

Frage: Warum hat ORG‑0 (θ=0) im Fenster [20,1e4] 133k Switches (Grid, k*-Jump) 
gegen ~85k Switches (Produktion, refine‑and‑verify)?  
→ Zeit‑Overlap, k‑Paar‑Kongruenz und Charakteristik der Überschüsse.

Design:
  [1] ORG‑0 Grid‑Switches regenerieren (wie in s27_e1b_orbit.py: k*-Sprung-Detektion).
  [2] Produktions‑Switches im selben Fenster laden.
  [3] Join: |t_grid − t_prod| < dt/2 = 0.001 → „Koinzidenz".
  [4] Kongruenz: gleiche k_left/k_right bei Koinzidenz → „Valid".
  [5] Nicht‑Matching Grid‑Switches charakterisieren:
      - adjacent‑Rate, ξ‑Residuen‑Verteilung, k_left‑Lage.

Ausgabe:
  - CSV: match_status, t_grid, t_prod, delta_t, k_grid, k_prod, adj, xi, verified.
  - Console: Statistiken (Overlap %, Adjacent‑Rate in Überschüssen, ξ‑Mediane).

Aufruf: python s28_org0_coincidence.py --switches zt_full_switches.csv.gz
Laufzeit: ~2–3 Min (Grid‑Walk + Join).
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import argparse
import csv
from pathlib import Path

import numpy as np

K_MIN, K_MAX = 3, 200
DT = 0.002
WIN_LO, WIN_HI = 20.0, 10_000.0
TIME_TOL = DT / 2.0          # 0.001 s Tolerance für Koinzidenz

# ---------------- Hilfsfunktionen ----------------
def sieve_primes(n_max):
    sieve = np.ones(n_max + 1, dtype=bool)
    sieve[:2] = False
    for i in range(2, int(n_max ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = False
    return sieve

def nu_matrix(primes, k_max):
    NU = np.zeros((k_max, len(primes)), dtype=np.int64)
    for j in range(2, k_max + 1):
        m = j
        for i, p in enumerate(primes):
            while m % p == 0:
                NU[j - 1, i] += 1
                m //= p
    return NU

def compute_walk_grid(t, NU):
    """Walk mit θ=0 (Originalbahn); k*-Jump-Detektion wie in E1b."""
    k_arr = np.arange(1, K_MAX + 1, dtype=float)
    weights = k_arr ** (-0.5)
    logs = np.log(k_arr)
    ksli = slice(K_MIN - 1, K_MAX)
    n = t.size
    depth = np.empty(n, dtype=np.float32)
    kstar = np.empty(n, dtype=np.int16)
    n_ch = (n + 20_000 - 1) // 20_000
    for c in range(n_ch):
        s, e = c * 20_000, min((c + 1) * 20_000, n)
        ang = -1j * t[s:e, None] * logs[None, :]
        P = np.cumsum(weights[None, :] * np.exp(ang), axis=1)
        Ps = P[:, ksli]
        absP = np.abs(Ps)
        idx = absP.argmin(axis=1)
        rows = np.arange(e - s)
        depth[s:e] = absP[rows, idx]
        kstar[s:e] = idx + K_MIN
        if (c + 1) % 25 == 0:
            print(f"    Chunk {c+1}/{n_ch} ({100*(c+1)/n_ch:.0f}%)", flush=True)
    return depth, kstar

def extract_grid_switches(kstar, t, time_tol):
    """K*-Jump-Detektion wie in s27_e1b_orbit.py."""
    dk = np.diff(kstar.astype(np.int32))
    sw_idx = np.where(dk != 0)[0]
    t_sw = t[sw_idx + 1]
    k_left = kstar[sw_idx]
    k_right = kstar[sw_idx + 1]
    adjacent = np.abs(k_left.astype(int) - k_right.astype(int)) == 1
    return t_sw, k_left, k_right, adjacent

def load_production_switches(path: Path, win_lo: float, win_hi: float):
    """Lade Produktions-Switches im Fenster (chunked, speicherschonend)."""
    cols = ["t_switch", "k_left", "k_right", "adjacent",
            "xi_residual", "verified"]
    dtypes = {"t_switch": np.float64, "k_left": np.int32,
              "k_right": np.int32, "adjacent": np.int8,
              "xi_residual": np.float64, "verified": np.int8}
    parts = []
    for chunk in pd.read_csv(path, compression="gzip", usecols=cols,
                             dtype=dtypes, chunksize=2_000_000):
        sel = (chunk["t_switch"] >= win_lo) & (chunk["t_switch"] <= win_hi)
        if sel.any():
            parts.append(chunk[sel])
    if not parts:
        return pd.DataFrame(columns=cols)
    return pd.concat(parts, ignore_index=True)

# ---------------- Hauptprogramm ----------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--switches", default="zt_full_switches.csv.gz")
    args = ap.parse_args()

    print("=" * 78)
    print("S28-ORG0-Koinzidenz: Grid (k*-Jump) vs. Produktion (refine)")
    print("=" * 78)

    # --- [1] ORG‑0 Grid‑Switches regenerieren ---
    n_grid = int((WIN_HI - WIN_LO) / DT) + 1
    t_grid = WIN_LO + DT * np.arange(n_grid)
    print(f"Grid-Fenster [{WIN_LO}, {WIN_HI:g}]: {n_grid:,} Punkte")

    sieve = sieve_primes(K_MAX)
    primes = sorted(p for p in range(2, K_MAX + 1) if sieve[p])
    NU = nu_matrix(primes, K_MAX)
    depth, kstar = compute_walk_grid(t_grid, NU)

    t_grid_sw, k_left_grid, k_right_grid, adj_grid = extract_grid_switches(
        kstar, t_grid, TIME_TOL)
    print(f"ORG‑0 Grid‑Switches (k*-Jump): {len(t_grid_sw):,}")

    # --- [2] Produktions-Switches laden ---
    prod_df = load_production_switches(Path(args.switches), WIN_LO, WIN_HI)
    print(f"Produktions-Switches im Fenster: {len(prod_df):,}")

    # --- [3] Zeit‑Join (Nearest Neighbor) ---
    print("\n[JOIN] Grid ↔ Produktion (Zeittoleranz ±0.001s)")
    prod_sorted = prod_df.sort_values("t_switch")
    t_prod = prod_sorted["t_switch"].values
    k_prod_left = prod_sorted["k_left"].values
    k_prod_right = prod_sorted["k_right"].values
    adj_prod = prod_sorted["adjacent"].values.astype(bool)
    xi_prod = prod_sorted["xi_residual"].values
    ver_prod = prod_sorted["verified"].values.astype(bool)

    # Für jeden Grid-Switch den nächsten Prod-Switch finden
    idx_prod = np.searchsorted(t_prod, t_grid_sw)
    # Kandidaten: links und rechts
    cand_left = np.clip(idx_prod - 1, 0, len(t_prod) - 1)
    cand_right = np.clip(idx_prod, 0, len(t_prod) - 1)
    dist_left = np.abs(t_grid_sw - t_prod[cand_left])
    dist_right = np.abs(t_grid_sw - t_prod[cand_right])
    choose_right = dist_right < dist_left
    idx_best = np.where(choose_right, cand_right, cand_left)
    delta_t = t_grid_sw - t_prod[idx_best]
    within_tol = np.abs(delta_t) <= TIME_TOL

    # --- [4] Kongruenz-Check (gleiche k-Paare) ---
    k_match = (k_left_grid == k_prod_left[idx_best]) & (
        k_right_grid == k_prod_right[idx_best])

    # --- [5] Charakterisierung ---
    out_rows = []
    n_overlap = 0
    n_congruent = 0
    adj_grid_only = []
    xi_grid_only = []
    k_grid_only = []

    for i, (t_g, k_l, k_r, adj_g, dtau, congru) in enumerate(zip(
            t_grid_sw, k_left_grid, k_right_grid, adj_grid, delta_t, k_match)):
        status = "none"
        if within_tol[i]:
            status = "overlap"
            if congru:                      # <— kein [i] mehr
                status = "congruent"
                n_congruent += 1
            n_overlap += 1
        else:
            # Grid-Only
            adj_grid_only.append(adj_g)
            xi_grid_only.append(np.nan)     # kein Prod-ξ für Grid-Only
            k_grid_only.append(k_l)

        out_rows.append({
            "grid_idx": i,
            "t_grid": t_g,
            "t_prod": t_prod[idx_best[i]],
            "delta_t_s": float(dtau),       # <— umbenannt, war 'dt'
            "k_left_grid": int(k_l),
            "k_right_grid": int(k_r),
            "k_left_prod": int(k_prod_left[idx_best[i]]),
            "k_right_prod": int(k_prod_right[idx_best[i]]),
            "adjacent_grid": int(adj_g),
            "adjacent_prod": int(adj_prod[idx_best[i]]) if within_tol[i] else None,
            "xi_prod": float(xi_prod[idx_best[i]]) if within_tol[i] else None,
            "verified_prod": int(ver_prod[idx_best[i]]) if within_tol[i] else None,
            "match_status": status,
            "kongruent": int(congru) if within_tol[i] else None,
        })

    # --- Statistik ---
    print("\n[ZUSAMMENFASSUNG]")
    print(f"   Grid‑Switches: {len(t_grid_sw):,} | Produkt.: {len(prod_df):,}")
    print(f"   Zeit‑Overlap: {n_overlap:,} ({100*n_overlap/len(t_grid_sw):.1f}%)")
    print(f"   Kongruente (gleiche k‑Paare): {n_congruent:,} ({100*n_congruent/len(t_grid_sw):.1f}%)")
    print(f"   Grid‑Only (Überproduktion): {len(adj_grid_only):,}")

    if len(adj_grid_only):
        print(f"   Grid‑Only adjacent‑Rate: {100*np.mean(adj_grid_only):.1f}%")
        print(f"   Grid‑Only k_left‑Median: {np.median(k_grid_only)}")

    # --- CSV ausgeben ---
    out = Path("s28_org0_coincidence.csv")
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_rows[0].keys())
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nCSV geschrieben: {out}")
    print("=" * 78)

if __name__ == "__main__":
    import faulthandler
    import sys
    import traceback
    faulthandler.enable()
    try:
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
        main()
    except BaseException:
        traceback.print_exc()
        input("\n[FEHLER] ENTER druecken ... ")