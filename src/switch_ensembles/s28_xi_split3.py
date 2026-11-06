#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
s28_xi_split3.py — S28: Finaler Film-Analyse-Schnitt (v3, sauber)

  [S1] Adjacent-ONLY Re-Cut: deep_pi & pol_M7, nur |Δk|=1, ξ-Quintile,
       Film je Quintil. Vorhersage: flach => Balance entlastet.
  [S2] 2D-Heatmap Δk-Bin × ξ-Quintil (deep_pi): Lokalisierung des Films.
       Δk-Bins: 1 | 2 | 3-4 | 5-8 | >8.  Quintile: global je Welt.
  [S3] Optional: Dissensus-Join — deep_pi-Events im Fenster [20,1e4]:
       Ist ein Event von einem ORG-0-Grid-Switch abgedeckt
       (t_event in [t_grid - dt, t_grid])? Deckungsrate je ξ-Quintil.

Alles in EINER Streaming-Passage: Auswahl, rel. ξ, L-Werte direkt am Event.
Aufruf:
  python s28_xi_split3.py --switches <pfad> \
      [--coincidence-csv s28_org0_coincidence.csv]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

K_MAX = 200
LN2 = np.log(2.0)
QCOEF = 1.0 / np.sqrt(2.0)

K_ARR = np.arange(1, K_MAX + 1, dtype=float)
WEIGHTS = K_ARR ** (-0.5)
LOGS = np.log(K_ARR)
HARM = np.cumsum(1.0 / K_ARR)
DYAD_M = np.arange(1, 9)
DYAD_W = QCOEF ** DYAD_M.astype(float)

DK_EDGES = [1.5, 2.5, 4.5, 8.5]       # -> Bins 1, 2, 3-4, 5-8, >8
DK_LABELS = ["1", "2", "3-4", "5-8", ">8"]
WIN_LO, WIN_HI = 20.0, 10_000.0

COLS = ["t_switch", "k_left", "k_right", "adjacent", "xi_residual"]

def world_from_phi(t):
    phi = np.degrees((t * LN2) % (2.0 * np.pi))
    dist0 = np.minimum(phi, 360.0 - phi)
    return (dist0 < 30.0), (dist0 >= 150.0)      # pol, pi

def eval_Lm_Lp(t_arr, kstar):
    """L_meas = |P_k* - S_dyad|, L_pred = |S_dyad|; chunked."""
    n = len(t_arr)
    Lm = np.empty(n); Lp = np.empty(n)
    CH = 50_000
    for s in range(0, n, CH):
        e = min(s + CH, n)
        tt = t_arr[s:e]; kk = kstar[s:e]
        ang = -1j * tt[:, None] * LOGS[None, :]
        P = np.cumsum(WEIGHTS[None, :] * np.exp(ang), axis=1)
        Pk = P[np.arange(e - s), kk - 1]
        M = np.floor(np.log2(kk)).astype(int)
        ph = -1j * tt[:, None] * (DYAD_M[None, :].astype(float) * LN2)
        S = np.where(DYAD_M[None, :] <= M[:, None],
                     DYAD_W[None, :] * np.exp(ph), 0.0).sum(axis=1)
        Lm[s:e] = np.abs(Pk - S)
        Lp[s:e] = np.abs(S)
    return Lm, Lp

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--switches",
                    default=r"C:\Users\k5d-q\OneDrive\Python\Zeta_PR\RELEVANT\beta_phase\session22workingdir\zt_full_switches.csv.gz")
    ap.add_argument("--coincidence-csv", default=None)
    ap.add_argument("--chunksize", type=int, default=2_000_000)
    args = ap.parse_args()

    print("=" * 78)
    print("S28-xi-Split v3: Adjacent-ONLY, Δk×ξ-Heatmap, Dissensus-Join")
    print("=" * 78)

    # ---------- Streaming: Welt-Auswahl + Werte direkt am Event ----------
    # Pro Welt akkumulieren: t, k*, xi_rel, dk, adj, Lm, Lp
    worlds = ("deep_pi", "pol_M7")
    D = {w: {k: [] for k in ("t", "ks", "xr", "dk", "adj", "Lm", "Lp")}
         for w in worlds}

    print("\n[P0] Streaming-Passage (Auswahl + L-Eval am Event) ...")
    n_seen = 0
    for chunk in pd.read_csv(args.switches, compression="gzip",
                             usecols=COLS,
                             dtype={"t_switch": np.float64,
                                    "k_left": np.int32,
                                    "k_right": np.int32,
                                    "adjacent": np.int8,
                                    "xi_residual": np.float64},
                             chunksize=args.chunksize):
        t = chunk["t_switch"].values
        kl = chunk["k_left"].values
        kr = chunk["k_right"].values
        adj = chunk["adjacent"].values.astype(bool)
        pol, pi = world_from_phi(t)
        m7 = kl >= 128
        xr = chunk["xi_residual"].values / HARM[kl - 1]
        dk = np.abs(kl - kr).astype(float)

        for wname, wm in (("deep_pi", pi & m7), ("pol_M7", pol & m7)):
            if not wm.any():
                continue
            tw = t[wm]; ksw = kr[wm]
            Lm, Lp = eval_Lm_Lp(tw, ksw)
            d = D[wname]
            d["t"].append(tw);      d["ks"].append(ksw)
            d["xr"].append(xr[wm]); d["dk"].append(dk[wm])
            d["adj"].append(adj[wm]); d["Lm"].append(Lm); d["Lp"].append(Lp)

        n_seen += len(chunk)
        print(f"    {n_seen:,}/14,673,327 ({100*n_seen/14_673_327:.0f}%)",
              flush=True)

    for w in worlds:
        D[w] = {k: np.concatenate(v) for k, v in D[w].items()}
        print(f"   {w}: n={len(D[w]['t']):,}")

    rows_out = []

    # ---------- [S1] Adjacent-ONLY Re-Cut ----------
    print("\n[S1] Adjacent-ONLY (|Δk|=1), ξ-Quintile, Film:")
    for w in worlds:
        d = D[w]
        am = d["adj"]
        print(f"  {w}: n_adjacent = {am.sum():,}")
        edges = np.quantile(d["xr"], [0.2, 0.4, 0.6, 0.8])  # globale Quintile
        qi = np.clip(np.searchsorted(edges, d["xr"], side="right"), 0, 4)
        film_all = d["Lm"] - d["Lp"]
        hdr = (f"   {'Quintil':>7} | {'n':>9} | {'mean_ξrel':>10} | "
               f"{'L_meas':>7} | {'L_pred':>7} | {'FILM':>8}")
        print(hdr); print("   " + "-" * (len(hdr) - 4))
        for qq in range(5):
            sel = am & (qi == qq)
            n = sel.sum()
            if n == 0:
                continue
            film = film_all[sel].mean()
            print(f"   {'Q'+str(qq+1):>7} | {n:>9,} | "
                  f"{d['xr'][sel].mean():>10.3e} | "
                  f"{d['Lm'][sel].mean():>7.4f} | {d['Lp'][sel].mean():>7.4f} "
                  f"| {film:>+8.4f}")
            rows_out.append({"cut": "adjacent_only", "world": w,
                             "quintil": f"Q{qq+1}", "n": int(n),
                             "film": f"{film:+.6f}"})

    # ---------- [S2] 2D-Heatmap Δk-Bin × ξ-Quintil ----------
    print("\n[S2] 2D-Heatmap Δk-Bin × ξ-Quintil (FILM je Zelle, deep_pi):")
    d = D["deep_pi"]
    edges = np.quantile(d["xr"], [0.2, 0.4, 0.6, 0.8])
    qi = np.clip(np.searchsorted(edges, d["xr"], side="right"), 0, 4)
    bi = np.clip(np.digitize(d["dk"], DK_EDGES), 0, 4)   # Bin 0..4
    film_all = d["Lm"] - d["Lp"]

    print(f"   {'Δk\\\\ξ':>6} | " + " | ".join(f"{'Q'+str(q+1):>7}" for q in range(5)))
    print("   " + "-" * 6 + "-+-" + "-+-".join(["-" * 7] * 5))
    for b in range(5):
        cells = []
        for qq in range(5):
            sel = (bi == b) & (qi == qq)
            if sel.any():
                cells.append(f"{film_all[sel].mean():>+7.3f}")
                rows_out.append({"cut": "heatmap", "world": "deep_pi",
                                 "quintil": f"Q{qq+1}",
                                 "dk_bin": DK_LABELS[b], "n": int(sel.sum()),
                                 "film": f"{film_all[sel].mean():+.6f}"})
            else:
                cells.append(f"{'—':>7}")
        print(f"   {DK_LABELS[b]:>6} | " + " | ".join(cells))

    # ---------- [S3] Dissensus-Join (optional) ----------
    if args.coincidence_csv:
        print(f"\n[S3] Dissensus-Join: {args.coincidence_csv}")
        try:
            co = pd.read_csv(args.coincidence_csv, usecols=["t_grid"])
            tg = np.sort(co["t_grid"].values)
            dt = 0.002
            d = D["deep_pi"]
            edges = np.quantile(d["xr"], [0.2, 0.4, 0.6, 0.8])
            qi = np.clip(np.searchsorted(edges, d["xr"], side="right"), 0, 4)
            win = (d["t"] >= WIN_LO) & (d["t"] <= WIN_HI)
            # Event ist abgedeckt, wenn ein Grid-Switch g existiert mit
            # t_event in [g - dt, g] (Intervall-Koinzidenz-Regel)
            idx = np.searchsorted(tg, d["t"], side="right") - 1
            cov = np.zeros(len(d["t"]), dtype=bool)
            ok = (idx >= 0)
            covered = ok & (d["t"] - tg[np.clip(idx, 0, len(tg)-1)] <= dt)
            cov = covered
            print(f"   Events im Fenster: {win.sum():,} | "
                  f"Deckung gesamt: {100*cov[win].mean():.1f}%")
            for qq in range(5):
                sel = win & (qi == qq)
                if sel.any():
                    print(f"   Q{qq+1}: Deckung {100*cov[sel].mean():>5.1f}% "
                          f"(n={sel.sum():,})")
                    rows_out.append({"cut": "coverage", "world": "deep_pi",
                                     "quintil": f"Q{qq+1}", "n": int(sel.sum()),
                                     "film": f"{100*cov[sel].mean():.2f}%covered"})
        except Exception as e:
            print(f"   WARN: Join fehlgeschlagen: {e}")

    # ---------- CSV ----------
    fields = sorted({k for r in rows_out for k in r}, )
    out = Path("s28_xi_split3.csv")
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows_out)
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