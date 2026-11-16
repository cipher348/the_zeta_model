#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
s27_e1a_surrogate.py — S27-E1a: Haar-Surrogat am Ort (Phasen-Randomisierung)

These: Struktur = Haar-Mass + argmin/Switch-Selektion.
Surrogat: Term j erhaelt Faktor exp(i*psi_j), psi_j = sum_p nu_p(j)*theta_p,
theta_p ~ U(0,2pi) iid ueber die Primzahlen <= 200, SAME Event-Zeiten.

Eigenschaften (per Konstruktion):
  - Betraege j^{-1/2} und Familienstruktur intakt
  - Einser: psi_1 = 0 -> bleibt der einzige festphasige Term
  - Dyadenleiter: psi_{2^m} = m*theta_2 -> bleibt koharent (Dyadenthron-Test!)
  - Komposit-Spiegel exakt erhalten (psi_4=2*theta_2, psi_6=theta_2+theta_3, ...)
  - Cross-Prime-Kohaerenzen & absolute Richtungen zerstoert (by design)

Auswertung in zwei Diagnose-Ebenen:
  [DEST] Rohes R@180 (sollte auf Uniform-Niveau fallen — Destruktions-
         Diagnostik nach s26_choir_match-Konvention)
  [ANCH] Prime-Chor relativ zum Dyaden-Anker (frame-invariant):
         anch_p = mean cos(phase_p - richtung(-S_dyad))

Seed 'EMP' = theta = 0 -> exakte empirische Referenz durch denselben
Code-Pfad (Pipline-QA gegen s26_ndeep_audit.csv moeglich).

Tests laut Spezifikation: T1 L-Profil, T2 Dyadenthron-Ratio, T3 Einser-
Rollen, T4 Prim-Ballast, T5 Last-Mediation, T6 zellweise Deckung,
V0 Anker-Validierung, QA Summe frac_F ~ 1.

Aufruf: python s27_e1a_surrogate.py --switches zt_full_switches.csv.gz --seeds 4
Laufzeit ~ (1+seeds) * s26_ndeep_audit.
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
FAM_NAMES = ["Einser", "Prim", "OddKomp", "EvenKomp"]
PROBES = [3, 5, 7, 11, 13]
APANEL = [3, 5, 13]
DYADS = {2 ** m for m in range(1, 8)}
LN2 = np.log(2.0)

# Profil-Schema (einzige Quelle fuer s27_e1a_profiles.csv)
KEYS = (["band_lo", "band_hi", "n", "L"] +
        [f"frac_{fn}" for fn in FAM_NAMES] +
        [f"a_{fn}" for fn in FAM_NAMES] +
        [f"frac_p{p}" for p in PROBES] +
        [f"a_p{p}" for p in APANEL] +
        [f"anch_p{p}" for p in PROBES])

# ---------------- Hilfsfunktionen ----------------
def sieve_primes(n_max):
    sieve = np.ones(n_max + 1, dtype=bool)
    sieve[:2] = False
    for i in range(2, int(n_max ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = False
    return sieve

def nu_matrix(primes, k_max):
    """NU[j-1, i] = nu_{p_i}(j) (Exponent von p_i in j)."""
    NU = np.zeros((k_max, len(primes)), dtype=np.int64)
    for j in range(2, k_max + 1):
        m = j
        for i, p in enumerate(primes):
            while m % p == 0:
                NU[j - 1, i] += 1
                m //= p
    return NU

def compute_walk(t: np.ndarray, psi: np.ndarray):
    """Walk mit Phasenverschiebung psi[j] (psi = 0 => Originalwalk)."""
    k_arr = np.arange(1, K_MAX + 1, dtype=float)
    weights = k_arr ** (-SIGMA)
    logs = np.log(k_arr)
    ksli = slice(K_MIN - 1, K_MAX)
    n = t.size
    depth = np.empty(n, dtype=np.float32)
    kstar = np.empty(n, dtype=np.int16)
    phat = np.empty(n, dtype=np.complex64)
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
        with np.errstate(invalid="ignore", divide="ignore"):
            phat[s:e] = (Ps[rows, idx] / absP[rows, idx]).astype(np.complex64)
        if (c + 1) % 25 == 0:
            print(f"    Chunk {c+1}/{n_ch} ({100*(c+1)/n_ch:.0f}%)", flush=True)
    return depth, kstar, phat

def R180(z, mask):
    """Rayleigh-R gegen 180 Grad (s26_choir_match-Konvention)."""
    zz = z[mask]
    return float(np.abs((-zz).mean())) if zz.size else float("nan")

# ---------------- Ein Durchgang (Referenz oder Surrogat) ----------------
def run_seed(t, theta, NU, sieve, primes, PIDX, label):
    print(f"\n--- {label}: Walk ---", flush=True)
    psi = NU.dot(theta)

    depth, kstar, phat = compute_walk(t, psi)

    # phi2 im Seed-Frame; dist0 ist konjugationsinvariant
    phi2 = np.degrees((-t * LN2 + theta[PIDX[2]]) % (2.0 * np.pi))
    dist0 = np.minimum(phi2, 360.0 - phi2)
    dq = np.quantile(depth, np.linspace(0, 1, 11))
    dbin = np.digitize(depth, dq[1:-1], right=False)
    deep = dbin <= 1                        # Tiefe NEU deziliert pro Seed
    band = np.clip((dist0 / 10.0).astype(int), 0, N_BANDS - 1)

    # Familien-Summen (nur aktive Terme j <= k*), psi-Shifts inklusive
    S_fam = {fn: np.zeros(t.size, dtype=np.complex64) for fn in FAM_NAMES}
    S_probe = {p: np.zeros(t.size, dtype=np.complex64) for p in PROBES}
    U_probe = {p: (np.exp(-1j * t * np.log(p)) *
                   np.exp(1j * theta[PIDX[p]])).astype(np.complex64)
               for p in PROBES}
    wts = np.arange(1, K_MAX + 1) ** (-SIGMA)

    for j in range(1, K_MAX + 1):
        if j >= 2 and j in DYADS:
            continue                          # Dyaden gehoeren nicht zu T
        term = (wts[j - 1] * np.exp(-1j * t * np.log(j)) *
                np.exp(1j * psi[j - 1])).astype(np.complex64)
        active = kstar >= j
        if j == 1:
            S_fam["Einser"] += term * active
        elif sieve[j] and j % 2 == 1:
            S_fam["Prim"] += term * active
            if j in S_probe:
                S_probe[j] += term * active
        elif j % 2 == 1:
            S_fam["OddKomp"] += term * active
        else:
            S_fam["EvenKomp"] += term * active
        if j % 50 == 0:
            print(f"    j={j}/{K_MAX}", flush=True)

    S_dyad = np.zeros(t.size, dtype=np.complex64)
    for m in DYADS:
        sel = kstar >= m
        S_dyad[sel] += (m ** (-SIGMA)) * np.exp(-1j * t[sel] * np.log(m)) \
                       * np.exp(1j * psi[m - 1])

    T = (depth * phat).astype(np.complex64) - S_dyad
    L = np.abs(T)
    ok = deep & (L > 1e-3)

    # ---- Bandprofile (s26_ndeep_audit-Konvention) ----
    bands_used = [b for b in range(N_BANDS) if (ok & (band == b)).sum() >= 200]
    prof = {k: [] for k in KEYS}              # <— KEYS statt lokalem keys
    fracsum = []
    for b in bands_used:
        m = ok & (band == b)
        lm = float(np.mean(L[m]))
        that = T[m] / np.maximum(L[m], 1e-3)
        prof["band_lo"].append(b * 10); prof["band_hi"].append(b * 10 + 10)
        prof["n"].append(int(m.sum())); prof["L"].append(lm)
        for fn in FAM_NAMES:
            sf = S_fam[fn][m]
            prof[f"frac_{fn}"].append(
                float(np.mean(np.real(sf * np.conj(that))) / lm))
            nrm = np.maximum(np.abs(sf), 1e-9)
            prof[f"a_{fn}"].append(
                float(np.mean(np.real((sf / nrm) * np.conj(that)))))
        fracsum.append(sum(prof[f"frac_{fn}"][-1] for fn in FAM_NAMES))
        for p in PROBES:
            sp = S_probe[p][m]
            prof[f"frac_p{p}"].append(
                float(np.mean(np.real(sp * np.conj(that))) / lm))
        for p in APANEL:
            sp = S_probe[p][m]
            nrm = np.maximum(np.abs(sp), 1e-9)
            prof[f"a_p{p}"].append(
                float(np.mean(np.real((sp / nrm) * np.conj(that)))))
        # ANCH: frame-invarianter Chor — Phase_p gegen Anti-Dyaden-Richtung
        dyd = S_dyad[m]
        anchor = -dyd / np.maximum(np.abs(dyd), 1e-9)
        for p in PROBES:
            prof[f"anch_p{p}"].append(
                float(np.mean(np.real(U_probe[p][m] * np.conj(anchor)))))

    # ---- DEST: rohe R@180 pro Welt (soll im Surrogat sterben) ----
    m_pol = deep & (dist0 < 30)
    m_pi = deep & (dist0 >= 150)
    diag = {"n_pol": int(m_pol.sum()), "n_pi": int(m_pi.sum())}
    for p in (3, 13):
        diag[f"R180_p{p}_pol"] = R180(U_probe[p], m_pol)
        diag[f"R180_p{p}_pi"] = R180(U_probe[p], m_pi)
    return prof, np.array(fracsum), bands_used, diag

# ---------------- Hauptprogramm ----------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--switches", default="zt_full_switches.csv.gz")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--rng", type=int, default=27)
    args = ap.parse_args()

    print("=" * 78)
    print("S27-E1a: Haar-Surrogat am Ort — Phasen-Randomisierung (theta_p)")
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
    print(f"Primzahlen <= {K_MAX}: {len(primes)} | NU {NU.shape}")

    rng = np.random.default_rng(args.rng)
    runs = []
    theta0 = np.zeros(len(primes))
    runs.append(("EMP", *run_seed(t, theta0, NU, sieve, primes, PIDX,
                                   "EMP (theta=0)")))
    for r in range(1, args.seeds + 1):
        theta = rng.uniform(0, 2 * np.pi, len(primes))
        runs.append((f"SUR-{r}",
                     *run_seed(t, theta, NU, sieve, primes, PIDX, f"SUR-{r}")))

    # ---------------- QA ----------------
    print("\n[QA] Summe frac_F pro Band (Abweichung von 1):")
    for name, prof, fs, bu, dg in runs:
        print(f"   {name:>6}: max|sum-1| = {np.max(np.abs(fs - 1.0)):.4f}"
              f"{' OK' if np.max(np.abs(fs - 1.0)) < 0.05 else '  <-- PRUEFEN!'}")

    bands = runs[0][3]
    idx_pol = [i for i, b in enumerate(bands) if b * 10 < 30]
    idx_pi = [i for i, b in enumerate(bands) if b * 10 >= 150]

    def col(name, key):
        return np.array(runs[[r[0] for r in runs].index(name)][1][key],
                        dtype=float)

    def sur_stack(key):
        return np.stack([col(n, key) for n, *_ in runs[1:]])

    # ---------------- Tests ----------------
    print("\n[V0] Anker-Validierung: Prim vs. Anti-Dyaden-Richtung (anch_p)")
    print("     (Empirie: falls 180-Grad-Lock = Anti-Dyaden-Lock, sollte")
    print("      anch_p(Pol) in Groessenordnung der R_p 0.35-0.45 liegen)")
    for p in (3, 13):
        a = col("EMP", f"anch_p{p}")
        ss = sur_stack(f"anch_p{p}")
        print(f"   p={p:<3} EMP pol={np.mean(a[idx_pol]):+.3f} "
              f"pi={np.mean(a[idx_pi]):+.3f} | SUR pol={ss[:, idx_pol].mean():+.3f}"
              f" pi={ss[:, idx_pi].mean():+.3f}")

    print("\n[DEST] Rohes R@180 (soll im Surrogat auf Uniform-Niveau fallen)")
    for name, prof, fs, bu, dg in runs:
        npol = max(dg["n_pol"], 1); npi = max(dg["n_pi"], 1)
        print(f"   {name:>6}: R3(pol)={dg['R180_p3_pol']:.3f} "
              f"[unif~{np.sqrt(np.pi/(4*npol)):.3f}]  "
              f"R13(pol)={dg['R180_p13_pol']:.3f}  "
              f"R3(pi)={dg['R180_p3_pi']:.3f}  "
              f"R13(pi)={dg['R180_p13_pi']:.3f}")

    print("\n[T1] Bandprofil L: Pearson(EMP, SUR-mean) ueber Bander")
    L_emp = col("EMP", "L"); L_sur = sur_stack("L")
    print(f"   Pearson = {np.corrcoef(L_emp, L_sur.mean(axis=0))[0,1]:.3f} "
          f"(Akzeptanz >= 0.90)")

    print("\n[T2] Dyadenthron-Ratio L(pol)/L(pi)")
    def ratio(vecs):
        return vecs[:, idx_pol].mean(axis=1) / vecs[:, idx_pi].mean(axis=1)
    r_emp = ratio(L_emp[None, :])[0]
    r_sur = ratio(L_sur)
    print(f"   EMP = {r_emp:.2f}  SUR = {r_sur.mean():.2f} "
          f"+/- {r_sur.std(ddof=1):.2f}  (geometrisch erwartet ~5.8)")

    print("\n[T3] Einser-Rolle: frac_Einser (pol: Saboteur / pi: Ueberschiesser)")
    for reg, nm in [(idx_pol, "pol"), (idx_pi, "pi ")]:
        e = col("EMP", "frac_Einser")[reg]
        s = sur_stack("frac_Einser")[:, reg].mean(axis=0)
        print(f"   {nm}: EMP med={np.median(e):+.3f} "
              f"(negativ in {np.sum(e < 0)}/{len(e)}) | "
              f"SUR med={np.median(s):+.3f} "
              f"(negativ in {np.sum(s < 0)}/{len(s)})")

    print("\n[T4] Prim-Ballast: frac_Prim (pol: + / pi: -)")
    for reg, nm in [(idx_pol, "pol"), (idx_pi, "pi ")]:
        e = col("EMP", "frac_Prim")[reg]
        s = sur_stack("frac_Prim")[:, reg].mean(axis=0)
        print(f"   {nm}: EMP med={np.median(e):+.3f} | "
              f"SUR med={np.median(s):+.3f}")

    print("\n[T5] Last-Mediation: Pearson(a_p, L) ueber Bandmittel")
    for p in APANEL:
        ae = col("EMP", f"a_p{p}"); se = sur_stack(f"a_p{p}")
        pe = np.corrcoef(ae, L_emp)[0, 1]
        ps = [np.corrcoef(se[i], L_sur[i])[0, 1]
              for i in range(se.shape[0])]
        print(f"   p={p:<3} EMP={pe:.3f}  SUR={np.mean(ps):.3f}"
              f"+/-{np.std(ps, ddof=1):.3f}  (Akzeptanz >= 0.80)")

    print("\n[T6] Zellweise Deckung frac_F (4 Familien x Baender)")
    zcount2 = zcount3 = ncells = 0
    for fn in FAM_NAMES:
        e = col("EMP", f"frac_{fn}"); s = sur_stack(f"frac_{fn}")
        sd = np.maximum(s.std(axis=0, ddof=1), 1e-6)
        z = np.abs((s.mean(axis=0) - e) / sd)
        zcount2 += int(np.sum(z > 2)); zcount3 += int(np.sum(z > 3))
        ncells += len(e)
    pct2 = 100.0 * zcount2 / ncells
    print(f"   Zellen |z|>2: {zcount2}/{ncells} ({pct2:.0f}%)  "
          f"|z|>3: {zcount3}/{ncells} ({100.0*zcount3/ncells:.0f}%)")
    print(f"   (Akzeptanz: <20% ueber 2sigma; Abweicher sind E1b-Input)")

    # ---------------- CSV ----------------
    out = Path("s27_e1a_profiles.csv")
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        hdr = ["run"] + KEYS                   # <— KEYS statt keys
        w.writerow(hdr)
        for name, prof, fs, bu, dg in runs:
            for i in range(len(bu)):
                w.writerow([name] + [f"{prof[k][i]:.4f}"
                                     if isinstance(prof[k][i], float)
                                     else prof[k][i] for k in KEYS])
    print(f"\nProfile geschrieben: {out}")

    out2 = Path("s27_e1a_tests.csv")
    with out2.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["test", "observable", "emp", "sur_mean", "sur_sd",
                    "verdict"])
        # T1
        c_T1 = np.corrcoef(L_emp, L_sur.mean(axis=0))[0, 1]
        w.writerow(["T1", "Pearson(L_band, L_band)", f"{c_T1:.4f}",
                    "", "", "confirm" if c_T1 >= 0.90 else
                    ("gray" if c_T1 >= 0.70 else "refute")])
        # T2
        w.writerow(["T2", "L(pol)/L(pi)",
                    f"{r_emp:.4f}", f"{r_sur.mean():.4f}",
                    f"{r_sur.std(ddof=1):.4f}",
                    "confirm" if abs(r_emp - r_sur.mean()) <
                    2 * max(r_sur.std(ddof=1), 1e-6) else "refute"])
        # T3 / T4
        for reg, nm, key in [(idx_pol, "pol", "frac_Einser"),
                             (idx_pi, "pi", "frac_Einser"),
                             (idx_pol, "pol", "frac_Prim"),
                             (idx_pi, "pi", "frac_Prim")]:
            e = np.median(col("EMP", key)[reg])
            s = sur_stack(key)[:, reg].mean(axis=1)
            w.writerow([f"T3/T4", f"{key}[{nm}]_median", f"{e:+.4f}",
                        f"{np.median(s):+.4f}",
                        f"{np.std(s, ddof=1):+.4f}", ""])
        # T5
        for p in APANEL:
            pe = np.corrcoef(col("EMP", f"a_p{p}"), L_emp)[0, 1]
            ps = [np.corrcoef(se[i], L_sur[i])[0, 1]
                  for i in range(sur_stack(f"a_p{p}").shape[0])]
            pm = np.mean(ps)
            w.writerow(["T5", f"Pearson(a{p}, L)", f"{pe:.4f}",
                        f"{pm:.4f}", f"{np.std(ps, ddof=1):.4f}",
                        "confirm" if pm >= 0.80 else
                        ("gray" if pm >= 0.60 else "refute")])
        # T6
        w.writerow(["T6", "frac cells |z|>2 (%)", "", f"{pct2:.1f}", "",
                    "confirm" if pct2 < 20.0 else "refute"])
        # V0 / DEST
        for p in (3, 13):
            a = col("EMP", f"anch_p{p}")
            ss = sur_stack(f"anch_p{p}")
            w.writerow(["V0", f"anch_p{p}_pol", f"{np.mean(a[idx_pol]):+.4f}",
                        f"{ss[:, idx_pol].mean():+.4f}", "", ""])
        for name, prof, fs, bu, dg in runs:
            w.writerow(["DEST", f"R180_p3_pol[{name}]",
                        f"{dg['R180_p3_pol']:.4f}", "", "", ""])

    print(f"Test-Zusammenfassung geschrieben: {out2}")

    # ---------------- Kurz-Verdict ----------------
    verdict = []
    verdict.append(("T1", c_T1))
    for p in APANEL:
        pass  # Einzelwerte bereits oben gedruckt
    verdict.append(("T6", pct2 / 100.0))
    print("\n" + "=" * 78)
    print("VERDICT-KURZ: T1 Pearson L =", f"{c_T1:.3f}",
          "| T6 Zellen>2sigma =", f"{pct2:.0f}%")
    print("Detail-Verdict je Test: s27_e1a_tests.csv (Spalte 'verdict')")
    print("Naechster Schritt bei Grauzone/Refutation: E1b "
          "(Bahnhofhoehen-Translate, Switch-konditioniert) — vgl. Spez.")
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