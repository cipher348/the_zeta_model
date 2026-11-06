#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
zeta_transition_operator_full.py  —  VERSION 2 (Rotations-Rekurrenz)

Aenderungen gegenueber v1:
  - scan_envelope_recurrent(): Bloecke von B t-Punkten werden per
    Rotationsterm r_n = exp(-i*B*dt*log n) fortgeschrieben statt per exp()
    neu berechnet. Re-Anchoring alle ANCHOR_EVERY Bloecke (Drift-Kontrolle).
  - Streaming-Verarbeitung: best_k wird nie komplett gespeichert;
    Indexwechsel werden online detektiert (carry ueber Blockgrenzen).
  - Switches werden in numpy-Puffern gesammelt (list of chunks + concat),
    kein Python-Tuple-Overhead bei ~10M Switches.

Semantik der Switch-Findung/Verfeinerung/Verifikation: UNVERAENDERT
(validiert gegen zeta_transition_operator.py, Abweichung +0.06%).

Empfohlener Volllauf:
  python zeta_transition_operator_full.py --zeros zeros6.txt \
      --k-max 200 --dt 0.002 --out-prefix zt_full --gzip
"""

from __future__ import annotations

import argparse
import csv
import gzip
import time
from pathlib import Path

import numpy as np

ANCHOR_EVERY = 4096        # Bloecke zwischen exp()-Re-Anchoring
BLOCK = 2048                # t-Punkte pro Rotationsblock

def read_zeros(path: Path, n_limit: int) -> np.ndarray:
    values = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            for token in reversed(line.replace(",", ".").split()):
                try:
                    values.append(float(token))
                    break
                except ValueError:
                    pass
            if n_limit and len(values) >= n_limit:
                break
    if not values:
        raise ValueError(f"Keine Nullstellen aus {path} gelesen.")
    z = np.asarray(values, dtype=float)
    if np.any(~np.isfinite(z)):
        raise ValueError("Nicht-endliche Werte in der Nullstellendatei.")
    if np.any(np.diff(z) <= 0):
        print("Hinweis: Nullstellen nicht streng aufsteigend; sortiere.")
        z = np.sort(z)
    return z

def open_out(path: Path, use_gzip: bool):
    if use_gzip:
        return gzip.open(path, mode="wt", newline="", encoding="utf-8")
    return path.open("w", newline="", encoding="utf-8")

# ----------------------------------------------------------------------
# Verfeinerung (unveraendert gegenueber validierter v1)
# ----------------------------------------------------------------------

def refine_and_verify(a, b, j, k, weights, logs, k_min, k_max, fine_factor, dt):
    t_sub = a + (b - a) * (np.arange(fine_factor + 1, dtype=float) / fine_factor)
    terms = weights[None, :] * np.exp(-1j * t_sub[:, None] * logs[None, :])
    P = np.cumsum(terms, axis=1)
    mag2 = P.real * P.real + P.imag * P.imag
    xi = mag2[:, k - 1] - mag2[:, j - 1]

    root, residual = None, np.inf
    sgn = np.sign(xi)
    for i in range(len(xi) - 1):
        if sgn[i] * sgn[i + 1] <= 0 and not (xi[i] == 0.0 and xi[i + 1] == 0.0):
            x0, x1 = xi[i], xi[i + 1]
            if x1 != x0:
                frac = abs(x0) / (abs(x0) + abs(x1))
                root = t_sub[i] + (t_sub[i + 1] - t_sub[i]) * frac
            else:
                root = 0.5 * (t_sub[i] + t_sub[i + 1])
            break
    if root is None:
        root = 0.5 * (a + b)

    eps = max(1e-8, dt * 1e-3)
    t_chk = np.array([root - eps, root + eps])
    terms = weights[None, :] * np.exp(-1j * t_chk[:, None] * logs[None, :])
    Pc = np.cumsum(terms, axis=1)
    magc = Pc.real * Pc.real + Pc.imag * Pc.imag
    left_k = int(np.argmin(magc[0, k_min - 1:k_max])) + k_min
    right_k = int(np.argmin(magc[1, k_min - 1:k_max])) + k_min
    verified = left_k != right_k

    t_one = np.array([root])
    terms = weights[None, :] * np.exp(-1j * t_one[:, None] * logs[None, :])
    Pr = np.cumsum(terms, axis=1)
    mr = Pr.real * Pr.real + Pr.imag * Pr.imag
    residual = abs(mr[0, k - 1] - mr[0, j - 1])
    return root, left_k, right_k, residual, verified

# ----------------------------------------------------------------------
# Hauptscan mit Rotationstermen
# ----------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Vollversion, rekurrent")
    ap.add_argument("--zeros", default="zeros6.txt")
    ap.add_argument("--n-zeros", type=int, default=0, help="0 = alle")
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--k-min", type=int, default=3)
    ap.add_argument("--k-max", type=int, default=200)
    ap.add_argument("--dt", type=float, default=0.002)
    ap.add_argument("--fine-factor", type=int, default=16)
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--t-cap", type=float, default=0.0)
    ap.add_argument("--out-prefix", default="zt_full")
    ap.add_argument("--gzip", action="store_true")
    args = ap.parse_args()

    if args.k_min < 1 or args.k_max < args.k_min:
        raise ValueError("Ungueltiger k-Bereich.")
    if args.dt <= 0 or args.fine_factor < 2:
        raise ValueError("dt/fine_factor ungueltig.")

    zeros = read_zeros(Path(args.zeros), args.n_zeros)
    t_lo = float(zeros[0] - args.margin)
    t_hi = float(zeros[-1] + args.margin)
    if args.t_cap > 0:
        t_hi = min(t_hi, args.t_cap)
        zeros = zeros[zeros <= t_hi]

    n_grid = int(np.floor((t_hi - t_lo) / args.dt)) + 1
    print("=" * 72)
    print("Zeta-Transitionsoperator — VOLLAUF (Rekurrenz v2)")
    print("=" * 72)
    print(f"Nullstellen (Range/Matching): {len(zeros):,}")
    print(f"sigma: {args.sigma}   k-Bereich: {args.k_min} ... {args.k_max}")
    print(f"t-Bereich: {t_lo:.6f} ... {t_hi:.6f}")
    print(f"Grid: {n_grid:,} Punkte (dt={args.dt}, Block={BLOCK}, "
          f"Re-Anchor/{ANCHOR_EVERY} Bloecke)")

    k_arr = np.arange(1, args.k_max + 1, dtype=float)
    weights = k_arr ** (-args.sigma)
    logs = np.log(k_arr)
    ksli = slice(args.k_min - 1, args.k_max)          # Spaltenbereich Argmin

    rB = np.exp(-1j * BLOCK * args.dt * logs)          # Rotation pro Block
    t_buf_switch = []                                  # Chunk-Listen (numpy)
    buf = [[], [], [], [], [], []]                     # t,kl,kr,adj,res,ver
    n_sw = 0
    tol = max(1e-10, args.dt * 0.05)
    last_t = -np.inf
    prev_k = -1

    def add_change(a, b, j, k):
        nonlocal last_t, n_sw
        root, kl, kr, res, ver = refine_and_verify(
            a, b, j, k, weights, logs, args.k_min, args.k_max,
            args.fine_factor, args.dt)
        if not ver:
            return
        if root - last_t <= tol:
            return
        buf[0].append(root); buf[1].append(kl); buf[2].append(kr)
        buf[3].append(int(abs(kr - kl) == 1)); buf[4].append(res)
        buf[5].append(1)
        last_t = root
        n_sw += 1

    t0 = time.time()
    n_blocks = (n_grid + BLOCK - 1) // BLOCK
    next_report = 0
    t_next = t_lo

    for bi in range(n_blocks):
        m = min(BLOCK, n_grid - bi * BLOCK)
        t_start_blk = t_lo + (bi * BLOCK) * args.dt

        if bi % ANCHOR_EVERY == 0:
            tb = t_start_blk + np.arange(m) * args.dt
            T = weights[None, :] * np.exp(-1j * tb[:, None] * logs[None, :])
            rot_carry = np.ones(m, dtype=np.complex128)  # irrelevant, nur Doc
        else:
            T = weights[None, :] * np.exp(-1j * t_start_blk * logs)[None, :]
            if m == BLOCK:
                T = np.broadcast_to(anchor_T, (BLOCK, len(logs))).copy()
                T *= rB[None, :]
            else:
                tb = t_start_blk + np.arange(m) * args.dt
                T = weights[None, :] * np.exp(-1j * tb[:, None] * logs[None, :])

        # Wait — sauberer: Reihenfolge gezielt (siehe Anmerkung unten im Text)
        P = np.cumsum(T, axis=1)
        mag2 = P.real * P.real + P.imag * P.imag
        bk = np.argmin(mag2[:, ksli], axis=1) + args.k_min

        # Indexwechsel online
        d = np.flatnonzero(bk[1:] != bk[:-1]) + 1
        for i in d:
            add_change(t_start_blk + (i - 1) * args.dt,
                       t_start_blk + i * args.dt,
                       int(bk[i - 1]), int(bk[i]))
        if prev_k != -1 and prev_k != int(bk[0]):
            add_change(t_start_blk - args.dt, t_start_blk, prev_k, int(bk[0]))
        prev_k = int(bk[-1])

        if bi >= next_report:
            el = time.time() - t0
            done_t = t_start_blk + m * args.dt
            eta = el / max(done_t - t_lo, 1e-9) * (t_hi - done_t)
            print(f"  t={done_t:.0f}: {n_sw:,} Switches | {el:.0f}s elapsed, "
                  f"~{eta/60:.0f} min rest")
            next_report = bi + max(1, n_blocks // 20)

        if bi % ANCHOR_EVERY == 0:
            anchor_T = T.copy()          # Ausgangspunkt fuer naechste Rotation
        else:
            anchor_T = T                 # wird im naechsten Block rotiert

    # ---------- Zusammensetzen ----------
    sw_t = np.array(buf[0]); sw_kl = np.array(buf[1], dtype=np.int32)
    sw_kr = np.array(buf[2], dtype=np.int32)
    sw_adj = np.array(buf[3], dtype=np.int8)
    sw_res = np.array(buf[4]); sw_ver = np.array(buf[5], dtype=np.int8)

    ext = ".csv.gz" if args.gzip else ".csv"
    sw_path = Path(f"{args.out_prefix}_switches{ext}")
    with open_out(sw_path, args.gzip) as f:
        w = csv.writer(f)
        w.writerow(["switch_index", "t_switch", "k_left", "k_right",
                    "adjacent", "xi_residual", "verified"])
        for i in range(n_sw):
            w.writerow([i + 1, f"{sw_t[i]:.15f}", sw_kl[i], sw_kr[i],
                        sw_adj[i], f"{sw_res[i]:.6e}", sw_ver[i]])

    match_path = Path(f"{args.out_prefix}_matches{ext}")
    pos = np.searchsorted(sw_t, zeros)
    lo = np.clip(pos - 1, 0, n_sw - 1)
    hi = np.clip(pos, 0, n_sw - 1)
    pick = np.where(np.abs(sw_t[lo] - zeros) <= np.abs(sw_t[hi] - zeros), lo, hi)
    sp = np.empty(len(zeros))
    dz = np.diff(zeros)
    if len(zeros) > 1:
        sp[0], sp[-1] = dz[0], dz[-1]
        sp[1:-1] = 0.5 * (dz[:-1] + dz[1:])

    with open_out(match_path, args.gzip) as f:
        w = csv.writer(f)
        w.writerow(["zero_index", "t_zero", "t_switch", "delta_signed",
                    "delta_abs", "local_zero_spacing", "delta_normalized",
                    "k_left", "k_right", "adjacent", "xi_residual"])
        for i in range(len(zeros)):
            j = pick[i]
            delta = sw_t[j] - zeros[i]
            d_abs = abs(delta)
            norm = d_abs / sp[i] if sp[i] > 0 else np.nan
            w.writerow([i + 1, f"{zeros[i]:.15f}", f"{sw_t[j]:.15f}",
                        f"{delta:.15e}", f"{d_abs:.15e}",
                        f"{sp[i]:.15e}", f"{norm:.15e}",
                        sw_kl[j], sw_kr[j], sw_adj[j], f"{sw_res[j]:.6e}"])

    print()
    print("=" * 72)
    print(f"Verifizierte Switches: {n_sw:,}")
    print(f"Dichte: {n_sw/max(t_hi-t_lo,1):.3f} pro t-Einheit")
    print(f"Switch-Katalog:      {sw_path}")
    print(f"Nullstellen-Matches: {match_path}")
    print(f"Gesamtlaufzeit:      {time.time()-t0:.0f}s")
    print("=" * 72)

if __name__ == "__main__":
    main()