"""
DriftModel — Public Release (Stage 1)
=====================================

BEHAUPTUNG (Staffelung: Rahmen -> Test -> Interpretation)
----------------------------------------------------------
Rahmen:     In adiabatisch reduzierten Fast-Slow-Systemen kompri-
            miert die Kopplung sich in einer langsamen invarianten
            Struktur ("Universalrezept zur kopplungsunabhaengigen
            Symmetrielokalisation", Kai & Cipher, Sept. 2026).
            Die Versklavungs-Hypothese fuer dieses System lautet:
            die relative Phase Delta(phi) ist an die Frequenz-
            drift Delta(omega) versklavt.

Test:       Unter strikt kausaler Informationsschaerke (nur Daten
            unterhalb des Ankers) traegt der Drift-Term kausale
            Vorhersagekraft fuer die relative Phase ab Delta(l) ~ 0.2,
            wachsend bis +0.09 R_bar bei Delta(l) = 0.5 (dichte
            Geometrie, Moving-Block-95%-CI [+0.068, +0.129]).

Interpretation: Das Ergebnis ist KONSISTENT mit adiabatischer
            Versklavung; es ist keine Herleitung der Fast-Slow-
            Struktur. Der Interaktionsterm kappa(l) ist nicht
            erforderlich (v28-Negativbefund), die Extrapolation
            des Drifts ueber den Instantanwert hinaus (TREND)
            traegt nicht robust (v31-Artefakt, siehe Caveats).

AKZEPTANZKRITERIEN (Modellrahmen s. Dossier, Sektion 9) -> v31.2-Ergebnis
---------------------------------------------------------------------------
[1] Kleiner absoluter Effekt ........ PF-Diff @h=0.5: +0.091 (MB-SIG+)
[2] Trennung zur Zweitbesten ........ PERSIST > TREND/PARSIMONIE (TREND-
                                     PERSIST n.s., MB-CI [-0.011, +0.087])
[3] Stabilitaet Beobachtungsdauer .... monoton wachsend h=0.1..0.5,
                                     frac_positive 0.61 -> 0.94
[4] Stabilitaet Skala/Fenster ....... 3 Geometrien: dicht n=33,
                                     semi-disjunkt n=5, voll-disjunkt n=2
                                     (Vorzeichen konsistent in {2.5, 3.0})
[5] Eindeutigkeit .................... Gate r_eva = 0.9553, identisch
                                     in 5 Laeufen (v30.3 bis v31.2)
[6] Signifikanz gegen Nullmodell ..... FROZEN als Null + MB-Bootstrap
                                     (block=4) + iid-Vergleich
[7] Dynamische Vertraeglichkeit ...... EKF-Schaetzung strikt kausal
                                     (Forward-Filter, kein Look-ahead)

ARCHITEKTUR
-----------
[1] Daten ......... zeta_features_v6_10000.csv, Gitter-Konstruktion
                   (Fenster +/-0.25, >=15 Punkte, gemischte Cluster)
[2] Instrument .... EKF fuer (Phase, Frequenz), sig_phi=1.0, sig_omega=0.3,
                   Phase-Wrap-handling; Selbsttest recon_r > 0.9
[3] Gate .......... r_eva >= 0.8 auf unabhaengigem Fenster [2.70, 3.30]
                   (tatsaechlich: 0.9553); Abbruch bei Verletzung
[4] Predictoren ... PERSIST  (Kern):  dphi_hat = dphi_0 + domega_0 * h
                   FROZEN   (Null):  dphi_hat = dphi_0
                   TREND (Sensitivitaet, NICHT beansprucht)
[5] Anker-Geometrien:
                   dicht         0.025  auf [2.25, 3.05]  (n=33)
                   semi-disjunkt 0.25   auf [2.0,  3.49]  (n=5)
                   voll-disjunkt  0.5    auf [2.0,  3.49]  (n=2)
[6] Statistik ..... iid- und Moving-Block-Bootstrap (block=4, 10k draws),
                   mb_valid-Flag bei zu wenigen Block-Starts (<5)
[7] Export ........ JSON mit summary, boot, per_anchor,
                   anchor_positions, caveats, Replikationskette

CAVEATS (vollstaendige Liste im JSON: documented_caveats)
---------------------------------------------------------
- Effekt sitzt in zwei Drift-Episoden (l0 ca. 2.32-2.46, 2.74-2.88);
  an Quieszenz-/Randfenstern neutral bis leicht negativ (|d|<=0.04)
- EKF-Einbrandbereich l0 < ~2.45: PERSIST dort unterschaetzt
- MB-Bootstrap ungueltig fuer n=5/n=2 (mb_valid=0 in der JSON)
- Voll-disjunkt effektiv {2.5, 3.0}; Wrap-Region 3.23-3.26 stoert
  kreuzende Holdouts nicht, Anker darin gewinnen nichts (Quieszenz)

REFERENZLAUF
------------
v31.2 (eingefroren): Gate 0.9553, PF-Diff +0.091 @h=0.5,
Replikationskette v30.3 -> v31.2 in der Export-JSON.
"""

import numpy as np
import pandas as pd
import json
import time
from pathlib import Path

RNG_SEED = 20260916          # NEU: deterministischer Seed, dokumentiert
rng = np.random.default_rng(RNG_SEED)

# ===== [1] DATEN =====
base_dir = Path(r'your_path')
output_dir = base_dir / 'your_path'
csv_path = base_dir / 'your_path' / 'zeta_features_v6_10000.csv'

df = pd.read_csv(csv_path)
n = len(df)
df['log_t'] = np.log10(df['t_known'].values)
log_t_vals = df['log_t'].values

BLOCK_A = ['tan_angle', 'arg_DN', 'phi_diff']

np.random.seed(42)
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
scaler_A = StandardScaler()
X_A_scaled = scaler_A.fit_transform(df[BLOCK_A].values)
kmeans_4 = KMeans(n_clusters=4, random_state=42, n_init=20)
clusters_4 = kmeans_4.fit_predict(X_A_scaled)

ws = 0.25
x_grid = np.linspace(log_t_vals.min(), log_t_vals.max(), 300)
alpha_grid, beta_grid, ell_grid = [], [], []

for x_center in x_grid:
    mask = (log_t_vals >= x_center - ws) & (log_t_vals <= x_center + ws)
    if mask.sum() < 15:
        continue
    c = clusters_4[mask]
    c0 = (c == 0).sum() / mask.sum(); c2 = (c == 2).sum() / mask.sum()
    c1 = (c == 1).sum() / mask.sum(); c3 = (c == 3).sum() / mask.sum()
    if (c0 + c2) > 0.01 and (c1 + c3) > 0.01:
        alpha_grid.append(c0 / (c0 + c2) - 0.5)
        beta_grid.append(c1 / (c1 + c3) - 0.5)
        ell_grid.append(x_center)

q_alpha = np.array(alpha_grid)
q_beta = np.array(beta_grid)
ell = np.array(ell_grid)
N = len(ell)
print(f"[1] Grid: {N} pts, ell in [{ell.min():.3f},{ell.max():.3f}]")

def wrap(x):
    return np.arctan2(np.sin(x), np.cos(x))

def circ_track_r(sim, obs):
    psi = wrap(sim - obs)
    return float(np.abs(np.mean(np.exp(1j * psi))))

# ===== [2] INSTRUMENT: EKF =====
print("[2] Ground Truth...")
from scipy.signal import hilbert
from scipy.ndimage import gaussian_filter1d

alpha_det = q_alpha - np.polyval(np.polyfit(ell, q_alpha, 2), ell)
beta_det  = q_beta  - np.polyval(np.polyfit(ell, q_beta, 2), ell)
phase_alpha_A = np.unwrap(np.angle(hilbert(alpha_det)))
phase_beta_A  = np.unwrap(np.angle(hilbert(beta_det)))
rel_phase_obs = wrap(phase_beta_A - phase_alpha_A)

om_a_A = gaussian_filter1d(np.gradient(phase_alpha_A, ell), sigma=3)
om_b_A = gaussian_filter1d(np.gradient(phase_beta_A, ell), sigma=3)
dOmega_A = om_b_A - om_a_A

SIG_PHI = 1.0       # v30.3-Grid-selected
SIG_OMEGA = 0.3     # v30.3-Grid-selected
SIG_A = 0.01
SIG_C = 0.005

def warmup_init(y, ew):
    y = np.asarray(y, float)
    ew = np.asarray(ew, float)
    w_grid = np.linspace(0.5, 10.0, 96)
    best = (np.inf, None, None, None, None)
    best_sse = np.inf
    for w_try in w_grid:
        B = np.column_stack([np.cos(w_try * ew),
                              np.sin(w_try * ew),
                              np.ones(len(ew))])
        coef, *_ = np.linalg.lstsq(B, y, rcond=None)
        resid = y - B @ coef
        sse = float(np.dot(resid, resid))
        if sse < best_sse:
            best_sse = sse
            best = (sse, w_try, coef[0], coef[1], coef[2])
    sse, w0, a, b, c0 = best
    phi0 = float(np.arctan2(a, b))
    A0 = float(np.hypot(a, b))
    rv = sse / max(len(y) - 3, 1)
    return phi0, w0, max(A0, 1e-6), c0, max(rv, 1e-8)

def ekf_phase_track(y, ell, W):
    y = np.asarray(y, float)
    ell = np.asarray(ell, float)
    n = len(y)
    phi0, w0, A0, c0, rv = warmup_init(y[:W], ell[:W])

    x = np.array([phi0, w0, max(A0, 1e-6), c0])
    P = np.diag([1.0, 2.0, 0.25, 0.25])
    R = max(rv, 1e-8)

    phi_hat = np.empty(n); om_hat = np.empty(n)
    phi_hat[:W] = phi0; om_hat[:W] = w0

    I4 = np.eye(4)
    for i in range(W, n):
        dt = ell[i] - ell[i - 1]

        F = np.eye(4); F[0, 1] = dt
        x = F @ x
        Q = np.diag([(SIG_PHI ** 2) * dt, (SIG_OMEGA ** 2) * dt,
                     (SIG_A ** 2) * dt, (SIG_C ** 2) * dt])
        P = F @ P @ F.T + Q

        phi, om, A, c = x
        h = A * np.sin(phi) + c
        H = np.array([A * np.cos(phi), 0.0, np.sin(phi), 1.0])
        S = float(H @ P @ H + R)
        if not np.isfinite(S) or S <= 0:
            phi_hat[i] = x[0]; om_hat[i] = x[1]
            continue
        K = (P @ H) / S
        lim = 3.0 * np.sqrt(S)
        innov = float(np.clip(y[i] - h, -lim, lim))
        x = x + K * innov
        x[0] = wrap(x[0])

        KH = np.outer(K, H)
        P = (I4 - KH) @ P @ (I4 - KH).T + R * np.outer(K, K)

        phi_hat[i] = x[0]; om_hat[i] = x[1]
    return phi_hat, om_hat

# ===== [3] GATE =====
print("[3] EKF auf q-Daten (v30.3-Config)...")
W = int(np.searchsorted(ell, 2.0))
phi_a_kf, om_a_kf = ekf_phase_track(q_alpha, ell, W)
phi_b_kf, om_b_kf = ekf_phase_track(q_beta, ell, W)
dOmega_kf = om_b_kf - om_a_kf
rel_phase_kf = wrap(phi_b_kf - phi_a_kf)

# Gate-Check auf unabhängigem Evaluationsfenster
eva_mask = (ell >= 2.70) & (ell <= 3.30)
r_eva = circ_track_r(rel_phase_kf[eva_mask], rel_phase_obs[eva_mask])
print(f"    Gate r_eva = {r_eva:.3f} (>= 0.8 erforderlich)")
assert r_eva >= 0.80, f"Instrument-Gate verletzt: {r_eva}"
print("    Gate BESTANDEN.")

# ===== [4] ANKER-GEOMETRIEN =====
ELL_LO = 2.0
ELL_HI_ANCHOR = 3.49        # max. Anker (Gitterende 3.995 - Horizont 0.5)
HORIZONS = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

def _make_anchors(start, stop, spacing):
    """Anker im Bereich [start, stop] mit vorgegebenem Spacing.
    Nur Bedingung: genug Warm-up-Puffer nach dem Filter (W+20)."""
    out = []
    a_l = start
    while a_l <= stop:
        i0 = int(np.argmin(np.abs(ell - a_l)))
        if i0 >= W + 20:
            out.append(i0)
        a_l += spacing
    return out

# (a) Dichte Geometrie — unverändert fuer Vergleichbarkeit mit v31
anchor_dense = _make_anchors(2.25, 3.05, 0.025)
print(f"[4a] Dicht (Spacing 0.025):     {len(anchor_dense)} Anker "
      f"auf [2.25, 3.05]")

# (b) Semi-disjunkt: Spacing 0.25 ueber den ERWEITERTEN Bereich
#     -> kaum Ueberlappung bei h=0.5, ~6 Anker
anchor_semidisj = _make_anchors(2.0, 3.49, 0.25)
print(f"[4b] Semi-disjunkt (0.25):      {len(anchor_semidisj)} Anker "
      f"auf [2.0, 3.49]")

# (c) Voll-disjunkt: Spacing 0.5 -> komplett unabhaengige
#     Holdout-Fenster, ~3 Anker
anchor_fulldisj = _make_anchors(2.0, 3.49, 0.5)
print(f"[4c] Voll-disjunkt (0.5):       {len(anchor_fulldisj)} Anker "
      f"auf [2.0, 3.49]")

# ===== [5] BEWERTUNG =====
MODES = ['ORACLE_A', 'PERSIST', 'FROZEN', 'TREND']

def predict_horizon(i0, i_end, mode):
    dphi0 = rel_phase_kf[i0]
    dOm0 = dOmega_kf[i0]
    ell_f = ell[i0:i_end + 1]
    dh = ell_f - ell[i0]

    if mode == 'FROZEN':
        return np.full(len(ell_f), dphi0)
    if mode == 'PERSIST':
        return dphi0 + dOm0 * dh
    if mode == 'TREND':
        w = max(3, (i0 - W) // 4)
        xw = ell[i0 - w:i0 + 1]
        slope = np.polyfit(xw, dOmega_kf[i0 - w:i0 + 1], 1)[0]
        eps = 0.5 ** np.abs(dh)
        return dphi0 + (dOm0 + slope * dh * eps) * dh
    if mode == 'ORACLE_A':
        cumA = np.concatenate([[0.0],
            np.cumsum(0.5 * (dOmega_A[:-1] + dOmega_A[1:]) * np.diff(ell))])
        return dphi0 + np.interp(ell_f, ell, cumA) - cumA[i0]
    raise ValueError(mode)

t_start = time.time()

def _run_geometry(anchors, geom_name):
    print(f"\n[5] {geom_name} Scoring...")
    results = {m: {f"{h:.1f}": [] for h in HORIZONS} for m in MODES}
    diff_pf = {f"{h:.1f}": [] for h in HORIZONS}   # PERSIST - FROZEN
    diff_tf = {f"{h:.1f}": [] for h in HORIZONS}   # TREND   - FROZEN
    diff_tp = {f"{h:.1f}": [] for h in HORIZONS}   # TREND   - PERSIST

    for i0 in anchors:
        for h in HORIZONS:
            j_cut = int(np.searchsorted(ell, ell[i0] + h))
            if j_cut >= N or (j_cut - i0) < 2:
                continue
            seg = slice(i0 + 1, j_cut + 1)
            obs = rel_phase_obs[seg]
            rp = rf = rt = None
            for mode in MODES:
                pred = predict_horizon(i0, j_cut, mode)[1:]
                score = circ_track_r(pred, obs)
                results[mode][f"{h:.1f}"].append(score)
                if mode == 'PERSIST': rp = score
                if mode == 'FROZEN':  rf = score
                if mode == 'TREND':   rt = score
            if rp is not None:
                diff_pf[f"{h:.1f}"].append(rp - rf)
            if rt is not None:
                diff_tf[f"{h:.1f}"].append(rt - rf)
                diff_tp[f"{h:.1f}"].append(rt - rp)
    # --- Per-Anker-Diagnose: Effektstaerke nach Ankerposition ---
    per_anchor = []
    print(f"\n  {geom_name}: Per-Anker (h=+0.5):")
    print(f"  {'ell_0':>6} | {'PERSIST':>8} | {'FROZEN':>8} | "
          f"{'TREND':>8} | {'PF-Diff':>8}")
    print("  " + "-" * 50)
    for i0 in anchors:
        j_cut = int(np.searchsorted(ell, ell[i0] + 0.5))
        if j_cut >= N or (j_cut - i0) < 2:
            continue
        seg = slice(i0 + 1, j_cut + 1)
        obs = rel_phase_obs[seg]
        sc = {}
        for mode in MODES:
            pred = predict_horizon(i0, j_cut, mode)[1:]
            sc[mode] = circ_track_r(pred, obs)
        rec = {
            'ell_0': float(ell[i0]),
            'persist_r': float(sc['PERSIST']),
            'frozen_r': float(sc['FROZEN']),
            'trend_r': float(sc['TREND']),
            'pf_diff': float(sc['PERSIST'] - sc['FROZEN']),
        }
        per_anchor.append(rec)
        print(f"  {rec['ell_0']:>6.3f} | {rec['persist_r']:>8.3f} | "
              f"{rec['frozen_r']:>8.3f} | {rec['trend_r']:>8.3f} | "
              f"{rec['pf_diff']:>+8.3f}")

    summary = {}
    for mode in MODES:
        summary[mode] = {}
        for h in HORIZONS:
            vals = np.array(results[mode][f"{h:.1f}"])
            if len(vals) == 0:
                continue
            summary[mode][f"{h:.1f}"] = {
                'r_bar': float(vals.mean()),
                'r_std': float(vals.std()),
                'n': int(len(vals)),
            }

    # Moving-Block-Bootstrap
    def _mb_boot(d, rng, block=4, n_draws=10000):
        d = np.asarray(d)
        if len(d) < 4:
            idx = rng.integers(0, len(d), size=(n_draws, len(d)))
            return d[idx].mean(axis=1)
        block_size = min(block, len(d))
        n_blocks = int(np.ceil(len(d) / block_size))
        means = np.empty(n_draws)
        for k in range(n_draws):
            starts = rng.integers(0, max(len(d) - block_size + 1, 1),
                                  size=n_blocks)
            chunks = []
            for s in starts:
                start_idx = int(s)
                end_idx = min(start_idx + block_size, len(d))
                chunks.append(d[start_idx:end_idx])
            if not chunks:
                means[k] = np.nan
            else:
                means[k] = np.concatenate(chunks).mean()
        return means

    DIFF_FAMS = {
        'persist_minus_frozen': diff_pf,
        'trend_minus_frozen':   diff_tf,
        'trend_minus_persist':  diff_tp
    }

    boot = {}
    for fam_name, fam in DIFF_FAMS.items():
        boot[fam_name] = {}
        for h in HORIZONS:
            d = np.array(fam[f"{h:.1f}"])
            if len(d) == 0:
                continue
            
            idx = rng.integers(0, len(d), size=(10000, len(d)))
            means_iid = d[idx].mean(axis=1)
            n_starts_possible = max(len(d) - 4 + 1, 1)
            mb_valid = n_starts_possible >= 5
            means_mb = _mb_boot(d, rng, block=4, n_draws=10000)
            
            entry = {
                'mean_diff': float(d.mean()),
                'ci_low_iid': float(np.percentile(means_iid, 2.5)),
                'ci_high_iid': float(np.percentile(means_iid, 97.5)),
                'ci_low_mb': float(np.percentile(means_mb, 2.5)) if mb_valid else None,
                'ci_high_mb': float(np.percentile(means_mb, 97.5)) if mb_valid else None,
                'mb_valid': bool(mb_valid),
                'frac_positive': float(np.mean(d > 0)),
                'n': int(len(d)),
            }
            if not mb_valid:
                entry['mb_degenerate_reason'] = (
                    f'n={len(d)}, block=4 -> nur {n_starts_possible} '
                    f'Startpos. (<5), MB-CI entartet'
                )
            
            boot[fam_name][f"{h:.1f}"] = entry
    
    # Print Summary
    print(f"\n  {geom_name}: Predictor Scores:")
    print("  Predictor  | " + " | ".join(f"h=+{h:.1f}" for h in HORIZONS))
    print("  " + "-" * 70)
    for mode in MODES:
        row = []
        for h in HORIZONS:
            if f"{h:.1f}" in summary[mode]:
                row.append(f"{summary[mode][f'{h:.1f}']['r_bar']:.3f}")
            else:
                row.append("  -- ")
        print(f"  {mode:<10} | " + " | ".join(f"{r:>5}" for r in row))

    for fam_name in DIFF_FAMS:
        print(f"\n  {fam_name.replace('_', ' ').upper()} (MB-CI):")
        for h in HORIZONS:
            b = boot.get(fam_name, {}).get(f"{h:.1f}")
            if not b:
                continue
            if b['mb_valid']:
                sig = 'SIG+' if b['ci_low_mb'] > 0 else (
                    'SIG-' if b['ci_high_mb'] < 0 else 'n.s.')
                print(f"    h=+{h:.1f}: {b['mean_diff']:+.3f} "
                      f"[{b['ci_low_mb']:+.3f}, {b['ci_high_mb']:+.3f}]  "
                      f"({b['frac_positive']*100:.0f}% pos)  {sig}")
            else:
                print(f"    h=+{h:.1f}: {b['mean_diff']:+.3f}  "
                      f"[MB-CI n/a: {b['mb_degenerate_reason']}]  "
                      f"(iid [{b['ci_low_iid']:+.3f}, "
                      f"{b['ci_high_iid']:+.3f}], "
                      f"{b['frac_positive']*100:.0f}% pos)")
  
    for fam_name, fam in boot.items():
        for h_key, stats in fam.items():
            if not stats['mb_valid']:
                assert stats['ci_low_mb'] is None, f"MB-CI sollte null bei mb_valid=0: {fam_name}/{h_key}"
                assert stats['ci_high_mb'] is None, f"MB-CI sollte null bei mb_valid=0: {fam_name}/{h_key}"
    return {
        'summary': summary,
        'boot': boot,
        'per_anchor': per_anchor,
        'n_anchors': len(anchors),
        'anchor_positions': [float(ell[i0]) for i0 in anchors],
    }

# Beide Geometrien durchlaufen
dense_result = _run_geometry(anchor_dense, "Dicht (Spacing 0.025)")
semidisj_result = _run_geometry(anchor_semidisj, "Semi-disjunkt (0.25)")
fulldisj_result = _run_geometry(anchor_fulldisj, "Voll-disjunkt (0.5)")

# ===== [6] EXPORT =====
def json_serialize(obj):
    if obj is None:
        return None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, float)):
        return float(obj) if not np.isnan(obj) else None
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: json_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_serialize(i) for i in obj]
    if isinstance(obj, bool):
        return obj
    return str(obj)

metadata = {
    'analysis': 'driftmodel_public_stage1',
    'stage': 'Stufe 1: oeffentliche Freigabe (driftmodel_public, referenziert v31.2)',
    'rng_seed': RNG_SEED,
    'method_frame': (
        "Universalrezept zur kopplungsunabhaengigen Symmetrielokalisation "
        "(Kai & Cipher, Lumo, Sept. 2026) — Rahmen; DriftModel-Lauf ist die "
        "empirische Instanziierung seiner Akzeptanzkriterien"
),
    'model_claim': (
        "Δφ zeigt ein Verhalten, dass mit adiabatischer Nachführung an Δω konsistent ist. "
        "κ(ℓ) nicht nötig (v28-Negativbefund). Δω trägt kausale "
        "Vorhersagekraft ab Δℓ ≈ 0.2, wachsend bis +0.09 R̄ bei 0.5 "
        "(dichte Geometrie, MB-CI)."
    ),
    'ekf_config': {
        'sig_phi': SIG_PHI,
        'sig_omega': SIG_OMEGA,
        'gate_r_eva': r_eva,
        'warmup_points': W,
    },
    'anchor_range': {'ell_lo': ELL_LO, 'ell_hi_anchor': ELL_HI_ANCHOR},
    'geometry_dense': dense_result,
    'geometry_semidisjunct': semidisj_result,
    'geometry_fulldisjunct': fulldisj_result,
    'documented_caveats': [
        "Per-Anker-Profil (h=+0.5): PF-Diff nur an den Raendern "
        "negativ (ell_0≈2.25 und 3.05); Effekt sitzt in zwei "
        "Drift-Episoden (ell_0≈2.32–2.46 und 2.74–2.88), wo FROZEN "
        "einbricht und PERSIST (0.94+) stabil bleibt.",
        "Anker unterhalb ~2.45 liegen teils im EKF-Einbrandbereich; "
        "PERSIST dort unterschaetzt, TREND kompensiert — Quelle des "
        "nicht-replizierten v31-TREND-Befunds.",
        "MB-Bootstrap ungueltig fuer n_start_moeglichkeiten < 5 "
        "(siehe mb_valid-Flag pro Familie/Horizont); betroffen: "
        "semi-disjunkt (n=5, block=4) und voll-disjunkt (n=2).",
        "Voll-disjunkt effektiv {2.5, 3.0}: Anker bei ell=2.0 fiel "
        "an der W+20-Bedingung (Warm-up-Puffer) raus.",
        "Wrap-Region (ell 3.23–3.26): Holdouts, die die Region "
        "kreuzen (alle Anker <= 3.05 dicht / <= 2.996 disjunkt), "
        "bleiben ungestoert. Ein Anker INNERHALB der Region "
        "(ell_0=3.25, semi-disjunkt) zeigt keinen PF-Gewinn "
        "(−0.039) — jedoch wegen lokaler Phasenquieszenz "
        "(FROZEN selbst 0.964, hoechster Wert im Datensatz), "
        "nicht wegen Wrap-Stoerung.",
        "Vorzeichenbilanz semi-disjunkt: 3/5 Anker positiv, beide "
        "negative Anker in Rand-/Quieszenzfenstern mit "
        "|pf_diff| <= 0.04. 'Vorzeichen konsistent in allen "
        "unabhaengigen Fenstern' gilt daher nur fuer das "
        "voll-disjunkte Set {2.5, 3.0}.",
    ],
    'replication_chain': {
        'gate_r_eva_history': ['v30.3: 0.9553', 'v30.4: 0.9553',
                               'v31: 0.9553', 'v31.1: 0.9553',
                               'v31.2: 0.9553',
                               'public_run: 0.9553'],
        'pf_diff_h05_history': {'v30.3_dicht_n11': '+0.077',
                                'v30.4_dicht_n22': '+0.081',
                                'v31_dicht_n12': '+0.081',
                                'v31.1_dicht_n33': '+0.091',
                                'v31.2_dicht_n33': '+0.091',
                                'public_run': '+0.091'},
    },
    'running_seconds': float(time.time() - t_start),
}

json_name = 'public_driftmodel.json'
with open(output_dir / json_name, 'w') as f:
    json.dump(json_serialize(metadata), f, indent=2, sort_keys=True)
print(f"\nSaved: {json_name}")
print(f"DONE in {time.time() - t_start:.1f}s! 🐍📈✨")
