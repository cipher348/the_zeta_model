#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
s30_grid_td.py — Session 30, T-D: Grid-Detektor-Replikation der Adjacent-Statistik
==================================================================================
Paper §8.3, offener Punkt 1. Zweck:

  Repliziert die Adjacent-pol/pi-Asymmetrie auf einem VOLLSTAENDIG
  unabhaengigen Detektor: naiver Argmin-Grid-Scanner, der P_k(t) auf einem
  aequidistanten Gitter direkt auswertet (Bohr-Darstellung), OHNE die
  Produktions-Pipeline (Interval-Refinement, Root-Interpolation, verify-Step).

  Erscheint die Asymmetrie auch hier, ist H3 (Detektor-Artefakt)
  endgueltig erledigt und T-D wandert im Paper von "pending" nach "done".

Design:
  - Gitter: dt = 0.002 (wie Produktions-Basisgitter), t in [t_lo, t_hi]
  - P_k(t) = sum_{n<=k} n^{-1/2} exp(-i t log n), k = 3..200
    chunked: Exp-Matrix (m, 200) + Cumsum entlang k -> |P_k| -> Argmin
  - Switch = Argmin-Aenderung zwischen aufeinanderfolgenden Gitterpunkten
  - Stamp = RECHTER Gitterpunkt (Interval-Kante, ORG-0-Konvention);
    dt-Fehler in phi_2 betraegt dt*ln2 ~ 0.08 Grad -> vernachlaessigbar
  - Auswertung analog s29_adjacent_geography: Welten (pol/mid/pi),
    18 Baender, M(k*), Sprungweiten-Klassen, M-gematchte Kontrolle,
    plus Inverse-Coupling-Pearson (adjacent vs |dk|>8) als Strukturcheck

Referenzanker (S29-Produktion, Fenster [20, 1e4]):
  n_total ~ 130.145 | pol 29.44% (n~5.8k) | pi 18.11% (n~26.5k) | mid 22.71%

REGISTRIERTE VERDICT-LOGIK (vorab, nicht nachtraeglich biegen!):
  - Sanity: n_switches in [78k, 182k] (+/-40% zur Produktion);
    ausserhalb -> WARNUNG, Verdict nur mit Vorbehalt
  - H3 REFUTED : raw-Ratio pol/pi >= 1.35
                (Asymmetrie repliziert auf unabhaengigem Detektor)
  - H3 ALIVE   : raw-Ratio <= 1.15
                (Asymmetrie verschwindet -> Pipeline-Artefakt-Verdacht)
  - GRAU       : 1.15 < ratio < 1.35
  - Strukturcheck: Pearson(adjacent-Rate, >8-Rate) ueber 18 Baender
    sollte die Inverse Kopplung reproduzieren (Produktion: -0.966)

PHASENFUNKTION: dist0_deg() ist VERBATIM aus s29_adjacent_geography.py
kopiert (Radiant-Modulo + E1d-Vorzeichen-Fix).
Fix-Backlog #8: Phasenfunktion NIEMALS rekonstruieren, immer kopieren.

Aufruf:
  python s30_grid_td.py --limit-chunks 3        # Smoke-Test (~150k Punkte)
  python s30_grid_td.py                         # Volllauf Fenster [20,1e4]
  python s30_grid_td.py --tmax 50000            # groesseres Fenster (optional)
  python s30_grid_td.py --dt 0.02 --limit-chunks 3   # schneller Smoke

Outputs:
  s30_grid_td.csv       (Long-Format: Welten/Kategorien, Baender, M-Zellen)
  s30_grid_td_log.txt   (alle Tabellen + Verdict)
"""

import argparse
import math
import sys
import time

import numpy as np
import pandas as pd

# --- Konstanten (konsistent zur S29-Serie) -------------------------------
LN2 = math.log(2.0)
K_MIN, K_MAX = 3, 200
POL_MAX, PI_MIN = 20.0, 150.0
M_MIN, M_MAX = 1, 7
MIN_CELL_N = 1000     # niedriger als S29 (Fruehfenster-Besetzung), im Log deklarieren
CATS = ["1", "2", "3-4", "5-8", ">8"]
WORLDS = ["pol", "pi", "mid"]
WIN_LO, WIN_HI = 20.0, 1.0e4

# Produktionsanker (S29, win-Frame [20,1e4], kstar=left)
PROD_ANCHOR = {"n_total": 130_145, "pol": 29.44, "pi": 18.11, "mid": 22.71}


def dist0_deg(t):
    """VERBATIM aus s29_adjacent_geography.py — nicht anfassen!"""
    deg = np.degrees(np.mod(-np.asarray(t, dtype=np.float64) * LN2, 2.0 * np.pi))
    return np.minimum(deg, 360.0 - deg)


def cat_index(dk):
    """|dk| -> Kategorie-Index 0..4 (wie s29_deficit_redistribution)."""
    idx = np.zeros(dk.shape, dtype=np.int8)
    idx[dk == 2] = 1
    idx[(dk >= 3) & (dk <= 4)] = 2
    idx[(dk >= 5) & (dk <= 8)] = 3
    idx[dk > 8] = 4
    return idx


def new_frame():
    return {
        "wc_n": np.zeros((3, 5), dtype=np.int64),   # Welt x Kategorie
        "MW_n": np.zeros((8, 3), dtype=np.int64),   # (M, Welt)
        "MW_a": np.zeros((8, 3), dtype=np.int64),
        "band_n": np.zeros(18, dtype=np.int64),
        "band_a": np.zeros(18, dtype=np.int64),
        "band_big": np.zeros(18, dtype=np.int64),
    }


def accumulate(fr, t_r, kl, kr):
    """Events (Arrays) in einen Frame-Akkumulator einrechnen."""
    if t_r.size == 0:
        return
    dk = np.abs(kr - kl)
    ci = cat_index(dk)
    adj = (dk == 1).astype(np.int64)
    big = (dk > 8)
    d0 = dist0_deg(t_r)
    widx = np.where(d0 <= POL_MAX, 0, np.where(d0 >= PI_MIN, 1, 2)).astype(np.int64)
    m = np.clip(np.floor(np.log2(kl)).astype(np.int64), M_MIN, M_MAX)
    band = np.clip((d0 // 10).astype(np.int64), 0, 17)

    key = widx * 5 + ci
    fr["wc_n"] += np.bincount(key, minlength=15).reshape(3, 5)
    np.add.at(fr["MW_n"], (m, widx), 1)
    np.add.at(fr["MW_a"], (m, widx), adj)
    fr["band_n"] += np.bincount(band, minlength=18)
    fr["band_a"] += np.bincount(band[dk == 1], minlength=18)
    fr["band_big"] += np.bincount(band[big], minlength=18)


def pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xm, ym = x - x.mean(), y - y.mean()
    den = math.sqrt(float((xm @ xm) * (ym @ ym)))
    return float((xm @ ym) / den) if den > 0 else float("nan")


def rate(n, a):
    return a / n if n > 0 else float("nan")


def world_rates(fr):
    out = {}
    for wi, w in enumerate(WORLDS):
        n = int(fr["wc_n"][wi].sum())
        a = int(fr["wc_n"][wi, 0])           # Kategorie 0 = adjacent
        out[w] = (n, a, rate(n, a))
    return out


def report(fr, log, label):
    log(f"\n===== FRAME [{label}] =====")
    wr = world_rates(fr)
    n_tot = sum(wr[w][0] for w in WORLDS)
    log(f"  n_total = {n_tot:,}")
    for w in WORLDS:
        n, a, r = wr[w]
        log(f"  {w:>4}: n={n:>9,}  adjacent={a:>9,}  rate={r*100:6.2f}%")
    if wr["pi"][2] > 0:
        log(f"  raw-Ratio pol/pi = {wr['pol'][2]/wr['pi'][2]:.3f}")

    # Sprungweiten-Shares je Welt
    log(f"  {'|dk|':>6} {'pol':>8} {'pi':>8} {'mid':>8}")
    for c in range(5):
        vals = []
        for wi in range(3):
            tot = fr["wc_n"][wi].sum()
            v = fr["wc_n"][wi, c] / tot * 100 if tot else float("nan")
            vals.append(v)
        log(f"  {CATS[c]:>6} " + " ".join(f"{v:8.3f}" for v in vals))

    # Band-Tabelle + Inverse-Coupling-Pearson
    with np.errstate(invalid="ignore", divide="ignore"):
        ar = np.where(fr["band_n"] > 0, fr["band_a"] / np.maximum(fr["band_n"], 1), np.nan)
        br = np.where(fr["band_n"] > 0, fr["band_big"] / np.maximum(fr["band_n"], 1), np.nan)
    log(f"  {'Band':>9} {'n':>12} {'adj %':>8} {'>8 %':>8}")
    for b in range(18):
        log(f"  {b*10:4d}-{b*10+10:<4d} {int(fr['band_n'][b]):12,} "
            f"{ar[b]*100:8.3f} {br[b]*100:8.3f}")
    ok = np.isfinite(ar) & np.isfinite(br)
    if ok.sum() > 2:
        p = pearson(ar[ok], br[ok])
        log(f"  Pearson(adjacent, >8) ueber {int(ok.sum())} Baender = {p:+.3f}"
            f"   (Produktion: -0.966)")

    # M-gematchte Kontrolle (pol vs pi)
    num = den_var = wsum = 0.0
    pol_num = pi_num = 0.0
    any_cell = False
    log(f"  {'M':>3} {'n_pol':>10} {'r_pol':>8} {'n_pi':>10} {'r_pi':>8} {'diff':>8}")
    for m in range(M_MIN, M_MAX + 1):
        n_po, a_po = int(fr["MW_n"][m, 0]), int(fr["MW_a"][m, 0])
        n_pi, a_pi = int(fr["MW_n"][m, 1]), int(fr["MW_a"][m, 1])
        if min(n_po, n_pi) < MIN_CELL_N or n_po == 0 or n_pi == 0:
            continue
        any_cell = True
        rp, ri = a_po / n_po, a_pi / n_pi
        d = rp - ri
        se = math.sqrt(max(rp * (1 - rp), 1e-12) / n_po +
                       max(ri * (1 - ri), 1e-12) / n_pi)
        w = n_pi
        num += w * d
        den_var += (w * se) ** 2
        wsum += w
        pol_num += w * rp
        pi_num += w * ri
        log(f"  {m:3d} {n_po:10,} {rp*100:8.3f} {n_pi:10,} {ri*100:8.3f} {d*100:+8.3f}")
    if any_cell and wsum > 0:
        rm = (pol_num / wsum) / (pi_num / wsum)
        z = num / math.sqrt(den_var) if den_var > 0 else float("nan")
        log(f"  M-gematchtes Ratio (Zellen n>={MIN_CELL_N}) = {rm:.3f} | z = {z:+.2f}")
        return rm
    log(f"  M-matching nicht durchfuehrbar (Zellen < n={MIN_CELL_N}).")
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tlo", type=float, default=20.0)
    ap.add_argument("--tmax", type=float, default=1.0e4)
    ap.add_argument("--dt", type=float, default=0.002)
    ap.add_argument("--chunk", type=int, default=50_000)
    ap.add_argument("--limit-chunks", type=int, default=None,
                    help="Smoke-Test: nur die ersten N Chunks verarbeiten")
    args = ap.parse_args()

    t0 = time.time()
    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(str(s))

    log(f"s30_grid_td | t in [{args.tlo}, {args.tmax}] | dt={args.dt} | "
        f"chunk={args.chunk:,} | limit-chunks={args.limit_chunks}")
    if args.dt != 0.002:
        log("WARNUNG: dt != 0.002 — Gitter weicht vom Produktions-Basisgitter ab,\n"
            "        Zahlen nur bedingt paper-relevant!")
    if args.limit_chunks is not None:
        log("WARNUNG: Smoke-Test-Modus — Zahlen NICHT paper-relevant!")

    logn = np.log(np.arange(1, K_MAX + 1, dtype=np.float64))
    amp = (1.0 / np.sqrt(np.arange(1, K_MAX + 1, dtype=np.float64)))

    n_points = int(round((args.tmax - args.tlo) / args.dt)) + 1
    n_chunks = (n_points + args.chunk - 1) // args.chunk
    if args.limit_chunks is not None:
        n_chunks = min(n_chunks, args.limit_chunks)
    log(f"[Grid] {n_points:,} Punkte in {n_chunks} Chunks "
        f"(k = {K_MIN}..{K_MAX})")

    fr_full = new_frame()
    fr_win = new_frame()
    n_switches = 0
    prev_k = None
    prev_t = None

    log("\n[Scan] Argmin-Gitterlauf...")
    for ci in range(n_chunks):
        i0 = ci * args.chunk
        i1 = min(i0 + args.chunk, n_points)
        if i0 >= i1:
            break
        t_chunk = args.tlo + np.arange(i0, i1, dtype=np.float64) * args.dt

        # P_k(t) chunked: Exp-Matrix + Cumsum entlang k
        E = np.exp(-1j * np.outer(t_chunk, logn))     # (m, 200)
        P = np.cumsum(E * amp[None, :], axis=1)       # P_k, k=1..200
        Psub = np.abs(P[:, K_MIN - 1:])               # k=3..200
        ks = np.argmin(Psub, axis=1) + K_MIN
        del E, P, Psub

        # Transitionen (inkl. Chunk-Grenze ueber prev_k/prev_t)
        if prev_k is None:
            seq_k = ks
            seq_t = t_chunk
            start = 1                      # erster Punkt: keine Historie
        else:
            seq_k = np.concatenate(([prev_k], ks))
            seq_t = np.concatenate(([prev_t], t_chunk))
            start = 0

        chg = seq_k[start + 1:] != seq_k[start:-1]
        idx = np.nonzero(chg)[0] + start    # Index des LINKEN PunkTS
        if idx.size:
            kl = seq_k[idx]
            kr = seq_k[idx + 1]
            t_r = seq_t[idx + 1]            # Stamp: rechter Gitterpunkt
            n_switches += idx.size
            accumulate(fr_full, t_r, kl, kr)
            wmask = (t_r >= WIN_LO) & (t_r <= WIN_HI)
            if wmask.any():
                accumulate(fr_win, t_r[wmask], kl[wmask], kr[wmask])

        prev_k = int(ks[-1])
        prev_t = float(t_chunk[-1])
        if (ci + 1) % 10 == 0 or ci + 1 == n_chunks:
            log(f"  ... Chunk {ci+1}/{n_chunks} | Switches bisher: {n_switches:,}"
                f" | {time.time()-t0:.1f}s")

    log(f"\n[Scan]fertig: {n_switches:,} Switches in {time.time()-t0:.1f}s")

    # --- Berichte ----------------------------------------------------------
    report(fr_full, log, f"full {args.tlo}:{args.tmax}")
    rm_win = None
    if args.tmax > WIN_HI or args.tlo > WIN_LO:
        rm_win = report(fr_win, log, f"win {WIN_LO:.0f}:{WIN_HI:.0f}")
    else:
        log("\n(Hinweis: Scan-Bereich == Referenzfenster, win-Frame identisch zu full.)")

    # --- Anker-Vergleich + VERDICT ----------------------------------------
    log("\n" + "=" * 72)
    log("T-D ANKER-VERGLEICH (win-Frame vs S29-Produktion)")
    log("=" * 72)
    wr = world_rates(fr_win if (args.tmax > WIN_HI or args.tlo > WIN_LO) else fr_full)
    n_tot = sum(wr[w][0] for w in WORLDS)
    log(f"  n_total  : Grid {n_tot:,} | Produktion ~{PROD_ANCHOR['n_total']:,}")
    for w in ("pol", "pi", "mid"):
        log(f"  rate {w:>3}: Grid {wr[w][2]*100:6.2f}% | "
            f"Produktion {PROD_ANCHOR[w]:6.2f}%")
    ratio = wr["pol"][2] / wr["pi"][2] if wr["pi"][2] > 0 else float("nan")

    sane_lo = int(PROD_ANCHOR["n_total"] * 0.6)
    sane_hi = int(PROD_ANCHOR["n_total"] * 1.4)
    n_sane = sane_lo <= n_tot <= sane_hi

    log("\n" + "=" * 72)
    log("VERDICT (registrierte Schwellwerte)")
    log("=" * 72)
    log(f"  Sanity n_switches in [{sane_lo:,}, {sane_hi:,}]: "
        f"{'OK' if n_sane else 'FEHLGESCHLAGEN — Vorbehalt!'}")
    log(f"  raw-Ratio pol/pi = {ratio:.3f}")
    if ratio >= 1.35:
        v = ("H3 REFUTED — die Adjacent-Asymmetrie repliziert sich auf dem\n"
             "  unabhaengigen Grid-Detektor. Kein Pipeline-Artefakt.\n"
             "  Paper: T-D von 'pending' nach 'done'.")
    elif ratio <= 1.15:
        v = ("H3 ALIVE — die Asymmetrie verschwindet auf dem unabhaengigen\n"
             "  Detektor. Pipeline-Artefakt-Verdacht, Untersuchung noetig.")
    else:
        v = ("GRAU — Asymmetrie teilweise repliziert. Detailanalyse noetig\n"
             "  (band-aufgeloest vergleichen, nicht nur Welt-Raten).")
    log(f"  -> {v}")
    if rm_win is not None:
        log(f"  (M-gematchtes Ratio im win-Frame: {rm_win:.3f}; "
            f"Produktion roh 1.63 im Fenster)")

    # --- Export ------------------------------------------------------------
    rows = []
    for fname, fr in (("full", fr_full), ("win", fr_win)):
        for wi, w in enumerate(WORLDS):
            tot = int(fr["wc_n"][wi].sum())
            for c in range(5):
                rows.append({"frame": fname, "world": w, "cat": CATS[c],
                             "n": int(fr["wc_n"][wi, c]),
                             "share": fr["wc_n"][wi, c] / tot if tot else float("nan")})
        for b in range(18):
            rows.append({"frame": fname, "world": "band", "cat": f"band{b:02d}",
                         "n": int(fr["band_n"][b]),
                         "share": fr["band_a"][b] / fr["band_n"][b]
                         if fr["band_n"][b] else float("nan")})
        for m in range(M_MIN, M_MAX + 1):
            for wi, w in enumerate(WORLDS):
                rows.append({"frame": fname, "world": f"M{m}", "cat": w,
                             "n": int(fr["MW_n"][m, wi]),
                             "share": fr["MW_a"][m, wi] / fr["MW_n"][m, wi]
                             if fr["MW_n"][m, wi] else float("nan")})
    pd.DataFrame(rows).to_csv("s30_grid_td.csv", index=False)

    with open("s30_grid_td_log.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    log(f"\n[Fertig] {time.time()-t0:.1f}s | s30_grid_td.csv + _log.txt")


if __name__ == "__main__":
    sys.exit(main())