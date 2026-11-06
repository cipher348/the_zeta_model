#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
s27_throne_check.py — S27: Thron-Ratio-Checkup (k*-Stufen-Mixtur)

These (nach E1a, T2-Luecke 4.60 vs 4.34): Der Thron-Ratio haengt an
der M(k*)-Verteilung der tiefen Events, M = floor(log2 k*). Reine
gerade M liefern konstant 5.83; ungerade Stufen daempfen stark.

Vorhersage OHNE Familien-Akkumulation (nur Walk + k*):
  L_pred(event) = | sum_{m=1..M(k*)} (1/sqrt2)^m * e^{i*m*phi2} |
  (kontinuierliche Dyadenleiter-Gain-Moduli; nur Walk noetig!)

Vergleich gegen:
  - empirisches L(b) aus s27_e1a_profiles.csv (run=EMP)
  - Thron-Ratio EMP (4.60) und SUR (4.34)

Outputs:
  [C1] Bandtabelle: n, M-Verteilung, L_pred vs L_emp
  [C2] Pearson(L_pred_band, L_emp_band)
  [C3] Ratio-Zerlegung: Mischung / rein-gerade / rein-ungerade
  [C4] Optional: Surrogat-Seeds -> M-Verteilungs-Stabilitaet
  CSV: s27_throne_check.csv

Aufruf:
  python s27_throne_check.py --switches zt_full_switches.csv.gz
  python s27_throne_check.py --switches ... --seeds 2  # mit M-Stabilitaet
Laufzeit: ~(1+seeds) * Walk-Passage (~2-3 min je Run), KEINE
Familien-Akkumulation.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

SIGMA = 0.5
K_MIN, K_MAX = 3, 200
CHUNK = 20_000
SUBSAMPLE = 2_000_000
N_BANDS = 18
LN2 = np.log(2.0)
Q = 1.0 / np.sqrt(2.0)          # Dyaden-Gewicht

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

def compute_walk(t: np.ndarray, psi: np.ndarray):
    """Wie E1a: Rueckgabe depth, kstar (phat hier unnoetig -> Float weg)."""
    k_arr = np.arange(1, K_MAX + 1, dtype=float)
    weights = k_arr ** (-SIGMA)
    logs = np.log(k_arr)
    ksli = slice(K_MIN - 1, K_MAX)
    n = t.size
    depth = np.empty(n, dtype=np.float32)
    kstar = np.empty(n, dtype=np.int16)
    n_ch = (n + CHUNK - 1) // CHUNK
    for c in range(n_ch):
        s, e = c * CHUNK, min((c + 1) * CHUNK, n)
        ang = (-1j * t[s:e, None] * logs[None, :]) + (1j * psi[None, :])
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

def ladder_gain(M_vals: np.ndarray, phi_deg: np.ndarray):
    """|sum_{m=1..M} q^m e^{i m phi}| je Event.
    M_vals: int-Array, phi_deg: Phase in Grad (0..180)."""
    out = np.zeros_like(M_vals, dtype=np.float64)
    ph = np.deg2rad(phi_deg)
    # maximale Stufe im Datensatz
    m_max = int(M_vals.max())
    acc = np.zeros_like(out)
    for m in range(1, m_max + 1):
        sel = M_vals >= m
        acc[sel] += (Q ** m) * np.cos(m * ph[sel])
        # Imaginaerteil separat: |sum| braucht Real+Imag
    # sauberer: Real- und Imag-Akkumulatoren
    re = np.zeros_like(out)
    im = np.zeros_like(out)
    for m in range(1, m_max + 1):
        sel = M_vals >= m
        re[sel] += (Q ** m) * np.cos(m * ph[sel])
        im[sel] += (Q ** m) * np.sin(m * ph[sel])
    return np.sqrt(re * re + im * im)

def run_pass(t, theta, NU, label):
    print(f"\n--- {label}: Walk (nur depth/k*, keine Familien) ---", flush=True)
    psi = NU.dot(theta)
    depth, kstar = compute_walk(t, psi)
    phi2 = np.degrees((-t * LN2 + (theta[0] if theta[0] else 0.0)) % (2.0 * np.pi))
    # dist0: konjugationsinvariant (cos(m*phi) identisch unter phi -> -phi)
    dist0 = np.minimum(phi2, 360.0 - phi2)
    dq = np.quantile(depth, np.linspace(0, 1, 11))
    dbin = np.digitize(depth, dq[1:-1], right=False)
    deep = dbin <= 1
    band = np.clip((dist0 / 10.0).astype(int), 0, N_BANDS - 1)
    M = np.floor(np.log2(np.maximum(kstar.astype(float), 2.0))).astype(int)
    return deep, band, dist0, M, depth

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--switches", default="zt_full_switches.csv.gz")
    ap.add_argument("--profiles", default="s27_e1a_profiles.csv")
    ap.add_argument("--seeds", type=int, default=0,
                    help="zusaetzliche Surrogat-Seeds fuer M-Stabilitaet")
    ap.add_argument("--rng", type=int, default=271)
    args = ap.parse_args()

    print("=" * 78)
    print("S27-Throne-Check: k*-Stufen-Mixtur des Thron-Ratios")
    print("=" * 78)

    import pandas as pd
    ts = []
    for chunk in pd.read_csv(args.switches, compression="gzip",
                             usecols=["t_switch"], dtype={"t_switch": np.float64},
                             chunksize=2_000_000):
        ts.append(chunk["t_switch"].to_numpy())
    t = np.concatenate(ts)
    t = t[np.argsort(t, kind="stable")]
    N = t.size
    stride = max(N // SUBSAMPLE, 1)
    t = t[np.arange(0, N, stride)]
    print(f"Subsample: {t.size:,} von {N:,} (Stride {stride})")

    sieve = sieve_primes(K_MAX)
    primes = [p for p in range(2, K_MAX + 1) if sieve[p]]
    PIDX = {p: i for i, p in enumerate(primes)}
    NU = nu_matrix(primes, K_MAX)

    # ---- Empirisches L(b) aus E1a-Profiles laden (run=EMP) ----
    L_emp = {}
    if Path(args.profiles).exists():
        df = pd.read_csv(args.profiles)
        dfe = df[df["run"] == "EMP"]
        for _, r in dfe.iterrows():
            L_emp[int(r["band_lo"])] = float(r["L"])
        print(f"Empirie geladen: {len(L_emp)} Baender aus {args.profiles}")
    else:
        print(f"WARNUNG: {args.profiles} nicht gefunden — "
              "nur interne Statistiken, kein L-Vergleich.")

    # ---- Runs: EMP (+ optionale Seeds) ----
    rng = np.random.default_rng(args.rng)
    runs = []

    theta0 = np.zeros(len(primes))
    deep, band, dist0, M, depth = run_pass(t, theta0, NU, "EMP (theta=0)")
    runs.append(("EMP", deep, band, dist0, M))

    for r in range(1, args.seeds + 1):
        theta = rng.uniform(0, 2 * np.pi, len(primes))
        th2 = theta[PIDX[2]]
        deep, band, dist0, M, depth = run_pass(t, theta, NU, f"SUR-{r}")
        runs.append((f"SUR-{r}", deep, band, dist0, M))

    out_rows = []

    for name, deep, band, dist0, M in runs:
        ok = deep
        print(f"\n[C1] {name}: Bandtabelle (M-Verteilung & L_pred)")
        print(f"   {'Band':>9} {'n':>7} {'Mmed':>5} {'f_gerade':>9} "
              f"{'L_pred':>8} {'L_emp':>8} {'diff%':>7}")
        pred_profile, emp_profile, n_used = [], [], []
        for b in range(N_BANDS):
            m = ok & (band == b)
            if m.sum() < 200:
                continue
            Mm = M[m]
            gm = ladder_gain(Mm, dist0[m])
            lp = float(np.mean(gm))
            le = L_emp.get(b * 10, np.nan)
            pred_profile.append(lp)
            emp_profile.append(le)
            n_used.append(int(m.sum()))
            fg = float(np.mean(Mm % 2 == 0))
            diff = 100.0 * (lp - le) / le if le == le else np.nan
            print(f"  {b*10:>3}-{b*10+10:<3} {m.sum():>7} "
                  f"{np.median(Mm):>5.0f} {fg:>9.3f} "
                  f"{lp:>8.3f} {le:>8.3f} "
                  f"{diff:>+7.1f}" if le == le else
                  f"  {b*10:>3}-{b*10+10:<3} {m.sum():>7} "
                  f"{np.median(Mm):>5.0f} {fg:>9.3f} {lp:>8.3f}   ---")
            out_rows.append([name, b * 10, b * 10 + 10, int(m.sum()),
                             float(np.median(Mm)), fg, lp,
                             f"{le:.4f}" if le == le else ""])

        pred_profile = np.array(pred_profile)
        emp_profile = np.array(emp_profile, dtype=float)
        sel = np.isfinite(emp_profile)

        # ---- C2: Pearson ----
        if sel.sum() > 3:
            pv = np.corrcoef(pred_profile[sel], emp_profile[sel])[0, 1]
            print(f"\n[C2] {name}: Pearson(L_pred, L_emp) = {pv:.3f}"
                  f"  (Akzeptanz >= 0.95)")

        # ---- C3: Ratio-Zerlegung ----
        idx_pol = np.array([(b * 10 < 30) for b in range(N_BANDS)])
        idx_pi = np.array([(b * 10 >= 150) for b in range(N_BANDS)])
        # auf benutzte Baelle reduzieren
        used_bands = [b for b in range(N_BANDS)
                      if ((ok & (band == b)).sum() >= 200)]
        m_pol_arr = np.array([b * 10 < 30 for b in used_bands])
        m_pi_arr = np.array([b * 10 >= 150 for b in used_bands])
        Lp_pol = pred_profile[m_pol_arr].mean()
        Lp_pi = pred_profile[m_pi_arr].mean()
        ratio_mix = Lp_pol / Lp_pi

        # Zerlegung: nur gerade M / nur ungerade M je Welt
        gm_all = ladder_gain(M[ok], dist0[ok])
        even_ok = ok & (M % 2 == 0)
        odd_ok = ok & (M % 2 == 1)
        for msk, lab in [(even_ok, "rein-gerade"),
                         (odd_ok, "rein-ungerade")]:
            if msk.sum() < 50:
                continue
            # Bandmittel nur in Weltbaendern
            pol_sel = msk & (dist0 < 30)
            pi_sel = msk & (dist0 >= 150)
            if pol_sel.sum() < 30 or pi_sel.sum() < 30:
                continue
            gp = ladder_gain(M[pol_sel], dist0[pol_sel]).mean()
            gq = ladder_gain(M[pi_sel], dist0[pi_sel]).mean()
            print(f"[C3] {name} {lab:>13}: G(pol)={gp:.3f} G(pi)={gq:.3f} "
                  f"ratio={gp/gq:.2f}  n_pol={pol_sel.sum():,} "
                  f"n_pi={pi_sel.sum():,}")

        print(f"[C3] {name} Mischung   : L_pred(pol)={Lp_pol:.3f} "
              f"L_pred(pi)={Lp_pi:.3f} ratio={ratio_mix:.2f}"
              f"   [Referenzen: EMP 4.60 | SUR 4.34 | Fixpunkt 5.83]")

        # ---- C4-Spur: M-Histogramm je Welt (Vergleichsdruck) ----
        h_pol = np.bincount(M[(ok) & (dist0 < 30)], minlength=10)
        h_pi = np.bincount(M[(ok) & (dist0 >= 150)], minlength=10)
        print(f"[C4] {name} M-Hist pol: {list(h_pol[:9])}")
        print(f"[C4] {name} M-Hist pi : {list(h_pi[:9])}")

    # ---------------- CSV ----------------
    out = Path("s27_throne_check.csv")
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run", "band_lo", "band_hi", "n", "M_median",
                    "frac_evenM", "L_pred", "L_emp"])
        for row in out_rows:
            w.writerow(row)
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