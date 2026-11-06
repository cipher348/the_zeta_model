#!/usr/bin/env python3
"""
s29_deficit_redistribution.py
=============================
Upgrade fuer Paper-Sektion 7.5: von "plausibly connected" zu "quantified".

FRAGE
-----
Ist das pi-DEFIZIT in der Adjacent-Rate das Spiegelbild der Grosssprung-
Dominanz tiefer pi-Events (S28/S3: 47.9% der deep-pi-Events haben |dk|>8)?
Also: Umverteilung entlang der Sprungweiten-Achse statt Aktivitaetsverlust?

TESTS
-----
T1  |dk|-Verteilung je Welt (full / win / M>=4-restricted), normiert.
T2  Komplementaritaet: Ist das Adjacent-Defizit (mid - pi) gleich dem
    Grosssprung-Exzess (pi - mid)? Plus die Gegenprobe fuer pol (sollte
    im Gleichgewicht ~0 sein, da rate(pol) ~ rate(mid)).
T3  Band-Aufloesung: je phi2-Band Adjacent-Rate vs Grosssprung-Rate ->
    Pearson ueber die 18 Baender. Generalisiert die Umverteilung von
    den zwei Welten auf den ganzen Phasenkreis.
T4  Konventions-Replikation via --kstar left|max.
T5  ANKER-CHECK: win-Frame roh pol/pi muss 29.4/18.1 reproduzieren,
    sonst ist die Konditionierung falsch und KEIN Verdict zaehlt.

VORAB REGISTRIERTE VERDICT-LOGIK (nicht nachtraeglich biegen!)
--------------------------------------------------------------
  A) |d_adj - d_big| / max(d_adj, d_big) <= 0.25  UND  Pearson(T3) <= -0.50
     -> REDISTRIBUTION CONFIRMED. 7.5 darf "quantified" schreiben:
        "das pi-Defizit ist der Einheitssprung-Schatten der Grosssprung-
         Dominanz; die pi-Welt springt nicht weniger, sie springt weiter."
  B) Pearson(T3) <= -0.50, aber Magnituden divergieren (>0.25)
     -> PARTIAL: Ordnungs-Inversion bestaetigt, Budget nicht geschlossen.
        Formulierung: "inverse Kopplung, kein geschlossenes Budget".
  C) sonst
     -> NOT SUPPORTED. 7.5 bleibt bei "plausibly connected" bzw. wird
        auf "unabhaengig" abgeschwaecht. Ehrlich berichten.

EHRLICHE EINSCHRAENKUNG
-----------------------
Adjacent (=1) und Grosssprung (>8) sind KEINE vollstaendigen Komplemente --
die Kategorien 2, 3-4, 5-8 liegen dazwischen. "Massenerhaltung" ist daher
approximativ und nur ueber die Randkategorien pruefbar. Deshalb wird die
VOLLE Verteilung mitberichtet (T1), damit sichtbar bleibt, wohin die Masse
tatsaechlich wandert.

FIX-CHANGE (nach S29_adjacent_geography.py-Review)
------------------------------------------------
Fehler: phi2 = mod(-t*ln2, 360) war falsch -- das Original verwendet
        mod(-t*ln2, 2*pi) in Radiant, dann degrees(...) umwandeln.
Fix: dist0_deg() jetzt exakt wie im Original (E1d-Vorzeichen-Fix + 
     Radiant-Modulo, nicht Grad-Modulo).
--> Anker-Check sollte jetzt 29.4/18.1 reproduzieren, alle Verdicts gültig.

AUFRUF
------
  python s29_deficit_redistribution.py --limit 500000   # Smoke-Test
  python s29_deficit_redistribution.py                  # kstar=left
  python s29_deficit_redistribution.py --kstar max      # Sensitivitaet

OUTPUT
------
  s29_deficit_redistribution_<kstar>.csv   (Long: block/frame/world/cat/n/share)
  s29_deficit_redistribution_<kstar>_bands.csv
  s29_deficit_redistribution_<kstar>_log.txt
"""

import argparse
import gzip
import math
import sys

import numpy as np
import pandas as pd

LN2 = math.log(2.0)
CATS = ["1", "2", "3-4", "5-8", ">8"]
WORLDS = ["pol", "pi", "mid"]
WIN_LO, WIN_HI = 20.0, 1.0e4
POL_MAX, PI_MIN = 20.0, 150.0


def cat_index(dk):
    """|dk| -> Kategorie-Index 0..4."""
    idx = np.zeros(dk.shape, dtype=np.int8)
    idx[dk == 2] = 1
    idx[(dk >= 3) & (dk <= 4)] = 2
    idx[(dk >= 5) & (dk <= 8)] = 3
    idx[dk > 8] = 4
    return idx


def dist0_deg(t):
    """EXAKT wie S29_adjacent_geography.py: phi2 in Radiant mod 2π, dann degrees().
    
    Dies ist die E1d-korrekte Version mit Vorzeichen-Fix (-t*ln2) und 
    Radiant-Modulo, nicht Grad-Modulo (BUG-FIX!).
    """
    deg = np.degrees(np.mod(-np.asarray(t, dtype=np.float64) * LN2, 2.0 * np.pi))
    return np.minimum(deg, 360.0 - deg)


def two_prop_z(a1, n1, a2, n2):
    """z-Test fuer zwei Anteile (gepoolte Varianz)."""
    if n1 == 0 or n2 == 0:
        return float("nan")
    p1, p2 = a1 / n1, a2 / n2
    p = (a1 + a2) / (n1 + n2)
    se = math.sqrt(p * (1.0 - p) * (1.0 / n1 + 1.0 / n2))
    return (p1 - p2) / se if se > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="zt_full_switches.csv.gz")
    ap.add_argument("--kstar", choices=["left", "max"], default="left")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--chunksize", type=int, default=2_000_000)
    args = ap.parse_args()

    prefix = f"s29_deficit_redistribution_{args.kstar}"
    log_lines = []

    def log(s=""):
        print(s)
        log_lines.append(str(s))

    log(f"s29_deficit_redistribution | kstar={args.kstar} | Limit {args.limit}")
    log(f"Datei: {args.path}")
    log(f"FIX: dist0_deg() jetzt exakt wie S29_adjacent_geography.py "
        f"(Radiant-Modulo, E1d-Fix)")
    log("")

    # --- Header sniffen ---------------------------------------------------
    try:
        with gzip.open(args.path, "rt") as fh:
            header = fh.readline().strip().split(",")
    except OSError as exc:
        sys.exit(f"[Fehler] Datei nicht lesbar: {exc}")

    need = ("k_left", "k_right", "t_switch")
    missing = [c for c in need if c not in header]
    if missing:
        sys.exit(f"[Fehler] Spalten fehlen: {missing}\nHeader: {header}")
    cols = list(need) + (["adjacent"] if "adjacent" in header else [])
    log(f"[Load] Spalten: {cols}")

    # --- Akkumulatoren ----------------------------------------------------
    acc = {}                      # (frame, world, cat) -> int
    band_n = np.zeros(18, dtype=np.int64)
    band_adj = np.zeros(18, dtype=np.int64)
    band_big = np.zeros(18, dtype=np.int64)

    n_rows = 0
    n_adj_col = 0
    n_adj_der = 0
    n_seen = 0

    reader = pd.read_csv(
        args.path, usecols=cols, chunksize=args.chunksize, compression="gzip"
    )

    for chunk in reader:
        t = chunk["t_switch"].to_numpy(dtype=np.float64)
        kl = chunk["k_left"].to_numpy(dtype=np.int64)
        kr = chunk["k_right"].to_numpy(dtype=np.int64)
        dk = np.abs(kr - kl)
        n_rows += t.size

        if "adjacent" in cols:
            n_adj_col += int(np.count_nonzero(chunk["adjacent"].to_numpy()))
            n_adj_der += int(np.count_nonzero(dk == 1))

        # phi2 mit E1d-Vorzeichen-Fix (RADIENT-MODULO, wie S29_adjacent_geography.py!)
        dist0 = dist0_deg(t)

        world = np.full(t.size, 2, dtype=np.int8)      # 2 = mid
        world[dist0 <= POL_MAX] = 0                    # 0 = pol
        world[dist0 >= PI_MIN] = 1                     # 1 = pi

        ks = kl if args.kstar == "left" else np.maximum(kl, kr)
        M = np.clip(np.floor(np.log2(np.maximum(ks, 1))).astype(np.int8), 1, 7)

        ci = cat_index(dk)
        band = np.minimum((dist0 // 10.0).astype(np.int8), 17)

        # --- Frames -------------------------------------------------------
        in_win = (t >= WIN_LO) & (t <= WIN_HI)
        masks = (
            ("full", np.ones(t.size, dtype=bool)),
            ("win", in_win),
            ("M4", M >= 4),
        )
        for frame, mask in masks:
            if not mask.any():
                continue
            key = world[mask].astype(np.int64) * 5 + ci[mask].astype(np.int64)
            cnt = np.bincount(key, minlength=15)
            for wi, wn in enumerate(WORLDS):
                base = wi * 5
                for cii, cn in enumerate(CATS):
                    k = (frame, wn, cn)
                    acc[k] = acc.get(k, 0) + int(cnt[base + cii])

        # --- Band-Akkumulation (Volllauf) ---------------------------------
        band_n += np.bincount(band, minlength=18)
        band_adj += np.bincount(band[ci == 0], minlength=18)
        band_big += np.bincount(band[ci == 4], minlength=18)

        n_seen += t.size
        if args.limit and n_seen >= args.limit:
            log(f"[Limit] Abbruch nach {n_seen:,} Events "
                f"-- Zahlen NICHT paper-relevant.")
            break

    log(f"[Load] {n_rows:,} Events gelesen.")
    if n_adj_col:
        log(f"[Check] adjacent-Spalte: {n_adj_col:,} | aus |dk|=1: {n_adj_der:,} "
            f"| Delta {abs(n_adj_col - n_adj_der):,}")
    log("")

    # --- T1/T2: Shares je Welt -------------------------------------------
    rows = []
    for frame in ("full", "win", "M4"):
        for wn in WORLDS:
            tot = sum(acc.get((frame, wn, c), 0) for c in CATS)
            for c in CATS:
                n = acc.get((frame, wn, c), 0)
                rows.append({
                    "block": "share", "frame": frame, "world": wn, "cat": c,
                    "n": n, "share": (n / tot) if tot else float("nan"),
                })
    df = pd.DataFrame(rows)

    log("=" * 72)
    log("T1 -- |dk|-VERTEILUNG JE WELT (Anteile in %)")
    log("=" * 72)
    for frame in ("full", "win", "M4"):
        sub = df[df.frame == frame]
        if sub.empty or sub["n"].sum() == 0:
            continue
        piv = sub.pivot(index="cat", columns="world", values="share").reindex(CATS)
        ntot = sub.groupby("world")["n"].sum().reindex(WORLDS)
        log(f"\n[{frame}]  n: " + "  ".join(f"{w}={int(ntot[w]):,}" for w in WORLDS))
        log(f"  {'|dk|':>6} {'pol':>8} {'pi':>8} {'mid':>8}")
        for c in CATS:
            vals = [piv.loc[c, w] * 100.0 for w in WORLDS]
            log(f"  {c:>6} " + " ".join(f"{v:8.3f}" for v in vals))

    # --- T2: Komplementaritaet -------------------------------------------
    log("")
    log("=" * 72)
    log("T2 -- KOMPLEMENTARITAET (Prozentpunkte)")
    log("=" * 72)

    comp_rows = []
    for frame in ("full", "M4"):
        sub = df[df.frame == frame]
        if sub.empty or sub["n"].sum() == 0:
            continue
        piv = sub.pivot(index="cat", columns="world", values="share").reindex(CATS)
        s1 = {w: piv.loc["1", w] * 100.0 for w in WORLDS}
        s8 = {w: piv.loc[">8", w] * 100.0 for w in WORLDS}

        d_adj = s1["mid"] - s1["pi"]          # pi-Defizit (adjacent)
        d_big = s8["pi"] - s8["mid"]          # pi-Exzess (Grosssprung)
        d_adj_pol = s1["pol"] - s1["mid"]     # Gegenprobe pol
        d_big_pol = s8["mid"] - s8["pol"]

        rel = (abs(d_adj - d_big) / max(d_adj, d_big)
               if max(d_adj, d_big) > 0 else float("nan"))

        # Randmassen-Summe je Welt: invariant -> reine Umverteilung
        edge_sum = {w: s1[w] + s8[w] for w in WORLDS}
        spread = max(edge_sum.values()) - min(edge_sum.values())

        log(f"\n[{frame}]")
        log(f"  pi-Defizit  (adjacent, mid-pi) = {d_adj:+7.3f} pp")
        log(f"  pi-Exzess   (>8,       pi-mid) = {d_big:+7.3f} pp")
        log(f"  relative Divergenz             = {rel:7.3f}   "
            f"(Schwelle A: <= 0.25)")
        log(f"  Gegenprobe pol: adjacent {d_adj_pol:+7.3f} pp | "
            f">8 {d_big_pol:+7.3f} pp")
        log(f"  Randmassen-Summe (adj+>8): pol {edge_sum['pol']:6.3f} | "
            f"pi {edge_sum['pi']:6.3f} | mid {edge_sum['mid']:6.3f} | "
            f"Spread {spread:.3f} pp")
        log(f"    -> Spread klein = Masse bleibt in den Randkategorien "
            f"(reine Umverteilung);")
        log(f"       Spread gross = Masse wandert in/aus den Mittelkategorien.")

        comp_rows.append({
            "block": "complementarity", "frame": frame, "world": "pi_vs_mid",
            "cat": "adjacent_deficit_pp", "n": 0, "share": d_adj,
        })
        comp_rows.append({
            "block": "complementarity", "frame": frame, "world": "pi_vs_mid",
            "cat": "large_excess_pp", "n": 0, "share": d_big,
        })

    # --- T3: Band-Aufloesung ---------------------------------------------
    log("")
    log("=" * 72)
    log("T3 -- BAND-AUFLOESUNG: Adjacent vs Grosssprung (Volllauf)")
    log("=" * 72)

    with np.errstate(invalid="ignore", divide="ignore"):
        adj_rate = np.where(band_n > 0, band_adj / band_n * 100.0, np.nan)
        big_rate = np.where(band_n > 0, band_big / band_n * 100.0, np.nan)

    log(f"  {'Band':>5} {'n':>12} {'adj %':>8} {'>8 %':>8}")
    for b in range(18):
        log(f"  {b*10:3d}-{b*10+10:3d} {band_n[b]:12,} "
            f"{adj_rate[b]:8.3f} {big_rate[b]:8.3f}")

    ok = np.isfinite(adj_rate) & np.isfinite(big_rate)
    pear = float(np.corrcoef(adj_rate[ok], big_rate[ok])[0, 1]) if ok.sum() > 2 \
        else float("nan")
    log(f"\n  Pearson(adjacent, >8) ueber {int(ok.sum())} Baender = {pear:+.3f}"
        f"   (Schwelle A/B: <= -0.50)")

    # --- T5: Anker-Check --------------------------------------------------
    log("")
    log("=" * 72)
    log("T5 -- ANKER-CHECK (win-Frame roh)")
    log("=" * 72)
    anchor_ok = False
    for wn in ("pol", "pi"):
        tot = sum(acc.get(("win", wn, c), 0) for c in CATS)
        a = acc.get(("win", wn, "1"), 0)
        r = a / tot * 100.0 if tot else float("nan")
        log(f"  {wn:>4}: {r:6.2f}%  (n={tot:,})")
    tot_pol = sum(acc.get(("win", "pol", c), 0) for c in CATS)
    tot_pi = sum(acc.get(("win", "pi", c), 0) for c in CATS)
    if tot_pol and tot_pi:
        rp = acc.get(("win", "pol", "1"), 0) / tot_pol * 100.0
        rq = acc.get(("win", "pi", "1"), 0) / tot_pi * 100.0
        anchor_ok = abs(rp - 29.4) < 1.5 and abs(rq - 18.1) < 1.5
        log(f"  Erwartet (S28): pol 29.4 / pi 18.1  ->  "
            f"{'OK' if anchor_ok else 'ABWEICHUNG -- Verdicts ungueltig!'}")

    # --- z-Test M4 --------------------------------------------------------
    log("")
    log("=" * 72)
    log("z-TEST (M>=4, adjacent pol vs pi)")
    log("=" * 72)
    npol = sum(acc.get(("M4", "pol", c), 0) for c in CATS)
    npi = sum(acc.get(("M4", "pi", c), 0) for c in CATS)
    apol = acc.get(("M4", "pol", "1"), 0)
    api = acc.get(("M4", "pi", "1"), 0)
    z = two_prop_z(apol, npol, api, npi)
    rpol = apol / npol * 100 if npol else float("nan")
    rpi = api / npi * 100 if npi else float("nan")
    log(f"  pol {rpol:.3f}% (n={npol:,}) | pi {rpi:.3f}% (n={npi:,}) | "
        f"z = {z:+.2f}")

    # --- VERDICT ----------------------------------------------------------
    log("")
    log("=" * 72)
    log("VERDICT")
    log("=" * 72)

    sub = df[df.frame == "full"]
    piv = sub.pivot(index="cat", columns="world", values="share").reindex(CATS)
    d_adj = (piv.loc["1", "mid"] - piv.loc["1", "pi"]) * 100.0
    d_big = (piv.loc[">8", "pi"] - piv.loc[">8", "mid"]) * 100.0
    rel = abs(d_adj - d_big) / max(d_adj, d_big) if max(d_adj, d_big) > 0 else 9.9

    if not anchor_ok:
        verdict = ("UNGUELTIG -- Anker-Check fehlgeschlagen, "
                   "Konditionierung pruefen (kein T4-Noetig)")
    elif rel <= 0.25 and pear <= -0.50:
        verdict = ("A) REDISTRIBUTION CONFIRMED -- pi-Defizit ist der "
                   "Einheitssprung-Schatten der Grosssprung-Dominanz "
                   "(7.5: quantified)")
    elif pear <= -0.50:
        verdict = ("B) PARTIAL -- Ordnungs-Inversion bestaetigt, Budget nicht "
                   "geschlossen (7.5: inverse Kopplung, kein geschlossenes "
                   "Budget)")
    else:
        verdict = ("C) NOT SUPPORTED -- Defizit ist keine einfache "
                   "Umverteilung (7.5 bleibt bei 'plausibly connected' "
                   "oder wird abgeschwaecht)")

    log(f"  Adjacent-Defizit {d_adj:+.3f} pp | Grosssprung-Exzess {d_big:+.3f} pp "
        f"| rel. Divergenz {rel:.3f}")
    log(f"  Pearson(Baender) {pear:+.3f}")
    log(f"  -> {verdict}")

    # --- Export -----------------------------------------------------------
    out = pd.concat([df, pd.DataFrame(comp_rows)], ignore_index=True)
    out.to_csv(f"{prefix}.csv", index=False)

    pd.DataFrame({
        "band": np.arange(18),
        "band_lo": np.arange(18) * 10,
        "band_hi": np.arange(18) * 10 + 10,
        "n": band_n,
        "n_adjacent": band_adj,
        "n_large": band_big,
        "rate_adjacent": adj_rate,
        "rate_large": big_rate,
    }).to_csv(f"{prefix}_bands.csv", index=False)

    with open(f"{prefix}_log.txt", "w") as fh:
        fh.write("\n".join(log_lines) + "\n")

    log("")
    log(f"[Fertig] {prefix}.csv + _bands.csv + _log.txt")


if __name__ == "__main__":
    main()
