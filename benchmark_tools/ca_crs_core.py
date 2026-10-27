"""
ca_crs_core.py
==============
Core scoring engine for CA-CRS+ (Cause-Aware Crowd Risk Score).

This file implements ONLY the deterministic math from the paper:
  - the non-linear risk score (Eq. 1-2)
  - causal attribution (Eq. 3)
  - cause-to-gate mapping + ripple-effect downgrade (Table II)
  - logarithmic resource demand (Eq. 4) + ADEQUATE/STRAINED/CRITICAL status
  - the GRS orchestrator (Eq. 6)
  - post-intervention projection (Eq. 7, Table III)

It does NOT do detection or optical flow. Counting (APGCC) and motion
(dense Farneback + conflict) are SEPARATE modules that feed (D, S, C) into
this engine. Keeping this pure makes it (a) trivially unit-testable and
(b) reusable for the sensitivity sweep WITHOUT re-running detection.

THREE CORRECTIONS baked in vs the published version (each flagged FIX#):
  FIX1  Saturation to [0,1]. The published Eq. (1)-(2) can exceed 3.0 at the
        crush state (D->1, S->0) because of gamma*exp(lambda*(D-S)), yet the
        printed SAFE/WARNING/DANGER bands assume [0,1]. The equations are only
        coherent with an explicit clamp. It is stated here; it must be stated
        in the paper.
  FIX2  GRS weighting is 2:1, not 3:1. Published Eq. (6) wrote
        omega = 2*I[DANGER] + 1  ->  weight 3 for DANGER, 1 otherwise (3:1),
        which contradicts the text ("DANGER zones weighted 2x") and the
        reference implementation (2:1). Reconcile text+equation+code to ONE
        choice. Default here: 2:1  (omega = I[DANGER] + 1).
  FIX3  Conflict = circular variance of flow directions, NOT the published
        "fraction opposing the dominant direction" (which collapses toward a
        meaningless value at a balanced 50/50 opposing split -- the exact case
        you must detect) and NOT the hackathon "IQR of head x-positions"
        (which measures spatial spread, not flow conflict). See conflict_index().

Author scaffolding for TENCON camera-ready rebuild. Verify every default
against your paper before use; nothing here is a substitute for your judgment.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import math
import numpy as np


# --------------------------------------------------------------------------- #
# Configuration (ALL values taken verbatim from the paper; re-confirm them,    #
# and let the Phase-2 sensitivity sweep justify the final weight choice).      #
# --------------------------------------------------------------------------- #
@dataclass
class CACRSConfig:
    # Scoring weights (paper: w1=0.4, w2=0.3, w3=0.3, gamma=0.05, lambda=4.0)
    w1: float = 0.4
    w2: float = 0.3
    w3: float = 0.3
    gamma: float = 0.05
    lam: float = 4.0

    # Risk bands (paper: SAFE < 0.35, WARNING 0.35-0.70, DANGER >= 0.70)
    warn_thr: float = 0.35
    danger_thr: float = 0.70

    # Attribution dominance threshold (paper: max r_f > 0.40 else MIXED)
    dominance_thr: float = 0.40

    # Conflict is deemed unreliable at very high density (paper: "~6 persons/m^2,
    # C_norm flagged unreliable, density drives attribution"). Operationalised
    # as D_norm >= 6/7 assuming 7 persons/m^2 crush density. CONFIGURABLE.
    conflict_unreliable_D: float = 6.0 / 7.0

    # Ripple-effect downgrade threshold (paper: downstream CA-CRS+ > 0.50)
    ripple_thr: float = 0.50

    # Resource demand coefficients by risk level (paper: alpha in {0,4,10})
    alpha_safe: float = 0.0
    alpha_warn: float = 4.0
    alpha_danger: float = 10.0

    # Resource status cut-offs on the infinity-norm of the shortfall vector
    # (paper: ADEQUATE if ||d||_inf == 0, STRAINED if <= 3, else CRITICAL)
    strained_max: float = 3.0

    # Post-intervention reduction coefficients (paper Table III), keyed by action
    xi: dict = field(default_factory=lambda: {
        "OPEN":     (0.60, 0.85, 0.90),   # (xi_d, xi_s, xi_c) exit gates
        "CLOSE":    (0.80, 0.55, 0.80),   # entry gates
        "REDIRECT": (0.85, 0.80, 0.40),   # flow zone
        "HOLD":     (0.95, 0.95, 0.95),   # hold / escalate
    })


class Risk(str, Enum):
    SAFE = "SAFE"
    WARNING = "WARNING"
    DANGER = "DANGER"


class Cause(str, Enum):
    DENSITY = "DENSITY"
    SPEED = "SPEED"
    CONFLICT = "CONFLICT"
    MIXED = "MIXED"
    SAFE = "SAFE"


# Cause -> primary gate action (paper Table II)
CAUSE_TO_GATE = {
    Cause.DENSITY:  "OPEN",
    Cause.SPEED:    "CLOSE",
    Cause.CONFLICT: "REDIRECT",
    Cause.MIXED:    "HOLD",
    Cause.SAFE:     "HOLD",
}


# --------------------------------------------------------------------------- #
# FIX3 : conflict index from a flow field (circular variance).                 #
# --------------------------------------------------------------------------- #
def conflict_index(angles: np.ndarray, magnitudes: np.ndarray | None = None) -> float:
    """
    Directional-conflict index in [0,1] from optical-flow vectors.

    C = 1 - Rbar,  where Rbar = |sum_i m_i * e^{i*theta_i}| / sum_i m_i

    Aligned flow      -> Rbar ~ 1 -> C ~ 0  (no conflict)
    Balanced opposing -> Rbar ~ 0 -> C ~ 1  (max conflict)  <-- does NOT self-cancel

    angles      : array of flow directions in radians
    magnitudes  : per-vector magnitudes (defaults to unit weights)
    """
    angles = np.asarray(angles, dtype=float).ravel()
    if angles.size == 0:
        return 0.0
    if magnitudes is None:
        magnitudes = np.ones_like(angles)
    magnitudes = np.asarray(magnitudes, dtype=float).ravel()
    total = magnitudes.sum()
    if total <= 1e-12:
        return 0.0
    resultant = np.abs(np.sum(magnitudes * np.exp(1j * angles))) / total
    return float(np.clip(1.0 - resultant, 0.0, 1.0))


# --------------------------------------------------------------------------- #
# Scoring (Eq. 1-2) with FIX1 saturation.                                      #
# --------------------------------------------------------------------------- #
def phi(D: float, S: float, cfg: CACRSConfig) -> float:
    """Non-linear penalty (Eq. 2): w2*S*(1-D) + gamma*exp(lambda*(D-S))."""
    return cfg.w2 * S * (1.0 - D) + cfg.gamma * math.exp(cfg.lam * (D - S))


def score(D: float, S: float, C: float, cfg: CACRSConfig) -> float:
    """
    CA-CRS+ risk score (Eq. 1) with FIX1 explicit saturation to [0,1].
    Inputs D, S, C are each assumed already normalised to [0,1].
    """
    D = float(np.clip(D, 0.0, 1.0))
    S = float(np.clip(S, 0.0, 1.0))
    C = float(np.clip(C, 0.0, 1.0))
    # FIX3-consistency: if density is in the crush band, conflict is unreliable
    # -> zero it so density drives both the score and the attribution.
    if D >= cfg.conflict_unreliable_D:
        C = 0.0
    raw = cfg.w1 * D + phi(D, S, cfg) + cfg.w3 * C
    return float(np.clip(raw, 0.0, 1.0))  # <-- FIX1


def classify(s: float, cfg: CACRSConfig) -> Risk:
    if s >= cfg.danger_thr:
        return Risk.DANGER
    if s >= cfg.warn_thr:
        return Risk.WARNING
    return Risk.SAFE


# --------------------------------------------------------------------------- #
# Attribution (Eq. 3) -> dominant cause.                                        #
# --------------------------------------------------------------------------- #
def attribute(D: float, S: float, C: float, cfg: CACRSConfig) -> tuple[Cause, dict]:
    """
    Dominant-factor attribution using LINEAR weighted ratios (Eq. 3).
    Returns (dominant_cause, {factor: ratio}). MIXED if no factor exceeds
    cfg.dominance_thr. Note: attribution is intentionally linear while the
    score is non-linear -- this must be stated explicitly in the paper.
    """
    if D >= cfg.conflict_unreliable_D:
        C = 0.0  # conflict unreliable in crush band
    wD, wS, wC = cfg.w1 * D, cfg.w2 * S, cfg.w3 * C
    denom = wD + wS + wC
    if denom <= 1e-12:
        return Cause.SAFE, {"DENSITY": 0.0, "SPEED": 0.0, "CONFLICT": 0.0}
    ratios = {"DENSITY": wD / denom, "SPEED": wS / denom, "CONFLICT": wC / denom}
    top_factor, top_ratio = max(ratios.items(), key=lambda kv: kv[1])
    if top_ratio <= cfg.dominance_thr:
        return Cause.MIXED, ratios
    return Cause[top_factor], ratios


# --------------------------------------------------------------------------- #
# Cause -> gate, with ripple-effect downgrade (Table II).                       #
# --------------------------------------------------------------------------- #
def gate_command(cause: Cause, downstream_score: float | None, cfg: CACRSConfig) -> dict:
    """
    Map a dominant cause to a gate action, then apply the ripple check:
    a REDIRECT is downgraded to HOLD iff the downstream receiving zone's
    CA-CRS+ exceeds cfg.ripple_thr. This is the paper's headline mechanism.
    """
    action = CAUSE_TO_GATE[cause]
    downgraded = False
    if action == "REDIRECT" and downstream_score is not None \
            and downstream_score > cfg.ripple_thr:
        action = "HOLD"
        downgraded = True
    return {"action": action, "ripple_downgraded": downgraded,
            "downstream_score": downstream_score}


# --------------------------------------------------------------------------- #
# Resource module (Eq. 4) + status.                                            #
# --------------------------------------------------------------------------- #
def marshal_demand(N: float, level: Risk, cfg: CACRSConfig) -> int:
    """Logarithmic personnel demand (Eq. 4): ceil(alpha_level * log10(1+N))."""
    alpha = {Risk.SAFE: cfg.alpha_safe,
             Risk.WARNING: cfg.alpha_warn,
             Risk.DANGER: cfg.alpha_danger}[level]
    return int(math.ceil(alpha * math.log10(1.0 + max(0.0, N))))


def resource_status(demands: dict, capacities: dict, cfg: CACRSConfig) -> dict:
    """
    demands, capacities: {resource_name: value}. Status is driven by the
    infinity-norm (max) of the positive shortfall vector, per the paper.
    NOTE: the paper's phrasing ("total shortfall across all zones") is slightly
    ambiguous about per-zone vs aggregate; this uses aggregate totals. Decide
    and document one convention.
    """
    shortfalls = {r: max(0.0, demands.get(r, 0.0) - capacities.get(r, 0.0))
                  for r in set(demands) | set(capacities)}
    inf_norm = max(shortfalls.values()) if shortfalls else 0.0
    if inf_norm == 0.0:
        status = "ADEQUATE"
    elif inf_norm <= cfg.strained_max:
        status = "STRAINED"
    else:
        status = "CRITICAL"
    return {"status": status, "shortfalls": shortfalls, "inf_norm": inf_norm}


# --------------------------------------------------------------------------- #
# GRS orchestrator (Eq. 6) with FIX2 (2:1 weighting).                           #
# --------------------------------------------------------------------------- #
def grs(scores: list[float], levels: list[Risk], cfg: CACRSConfig) -> float:
    """
    Urgency-weighted Global Risk Score. FIX2: omega = I[DANGER] + 1 (2:1).
    The published Eq. (6) used 2*I+1 (3:1); that contradicts the '2x' text.
    """
    omegas = [ (1.0 + (1.0 if lv == Risk.DANGER else 0.0)) for lv in levels ]  # 2 or 1
    num = sum(o * s for o, s in zip(omegas, scores))
    den = sum(omegas)
    return float(num / den) if den > 0 else 0.0


# --------------------------------------------------------------------------- #
# Post-intervention projection (Eq. 7, Table III).                             #
# --------------------------------------------------------------------------- #
def project_post_intervention(D: float, S: float, C: float,
                              action: str, cfg: CACRSConfig) -> float:
    """
    Projected CA-CRS+ AFTER a gate action, by scaling the features with the
    reduction coefficients (xi_d, xi_s, xi_c) for that action, then re-scoring.
    HARSH REMINDER: this projection is only as trustworthy as Table III. Those
    coefficients are currently ASSUMED. Ground them from a pedestrian
    fundamental-diagram argument (Phase 4) or label CRR as an estimate.
    """
    xd, xs, xc = cfg.xi[action]
    return score(xd * D, xs * S, xc * C, cfg)


# --------------------------------------------------------------------------- #
# Self-test: run `python ca_crs_core.py` to see the fixes work, no data needed. #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    cfg = CACRSConfig()

    def report(name, D, S, C, downstream=None):
        s = score(D, S, C, cfg)
        lv = classify(s, cfg)
        cause, ratios = attribute(D, S, C, cfg)
        gate = gate_command(cause, downstream, cfg)
        print(f"\n[{name}]  D={D:.2f} S={S:.2f} C={C:.2f}")
        print(f"   score={s:.3f}  level={lv.value}  cause={cause.value}")
        print(f"   ratios={{'D':{ratios['DENSITY']:.2f},"
              f"'S':{ratios['SPEED']:.2f},'C':{ratios['CONFLICT']:.2f}}}")
        print(f"   gate={gate['action']}"
              + (f"  (REDIRECT downgraded via ripple, downstream={downstream})"
                 if gate["ripple_downgraded"] else ""))

    print("=" * 68)
    print("CA-CRS+ core self-test (synthetic inputs; verifies the 3 fixes)")
    print("=" * 68)

    # 1) Crush: high density, near-zero speed -> exp term explodes, FIX1 clamps.
    raw_crush = cfg.w1*0.95 + phi(0.95, 0.05, cfg)  # conflict zeroed in crush band
    print(f"\nFIX1 demo -- crush state raw (unclamped) score would be "
          f"{raw_crush:.2f} (>1!). Clamped below:")
    report("crush", D=0.95, S=0.05, C=0.30)          # C ignored: density in crush band

    # 2) Panic speed -> SPEED dominant -> CLOSE entry gates.
    report("panic_speed", D=0.40, S=0.95, C=0.10)

    # 3) Opposing flow -> CONFLICT dominant -> REDIRECT, then ripple downgrade.
    report("opposing_flow", D=0.50, S=0.40, C=0.90, downstream=0.62)

    # FIX3 demo: conflict index does not self-cancel on a 50/50 opposing field.
    uni = np.zeros(200)                               # all flowing right
    opp = np.concatenate([np.zeros(100), np.full(100, math.pi)])  # 50/50 opposing
    print(f"\nFIX3 demo -- conflict_index: aligned flow = "
          f"{conflict_index(uni):.3f} (want ~0);  50/50 opposing = "
          f"{conflict_index(opp):.3f} (want ~1)")

    # FIX2 demo: GRS 2:1 vs the buggy 3:1.
    scores = [0.95, 0.40, 0.30]
    levels = [Risk.DANGER, Risk.WARNING, Risk.SAFE]
    correct = grs(scores, levels, cfg)
    buggy = (sum((2*(1 if l == Risk.DANGER else 0)+1)*s
                 for l, s in zip(levels, scores))
             / sum(2*(1 if l == Risk.DANGER else 0)+1 for l in levels))
    print(f"\nFIX2 demo -- GRS on {scores}: correct 2:1 = {correct:.3f}  "
          f"vs published 3:1 = {buggy:.3f}")

    # Resource module sanity check (matches paper-style numbers).
    dmd = marshal_demand(973, Risk.DANGER, cfg)       # paper example ~30
    print(f"\nResource demo -- marshals for N=973 @ DANGER = {dmd} "
          f"(paper-style example ~30)")
    print(resource_status({"marshals": 40}, {"marshals": 20}, cfg))
    print("\nOK. If these read sensibly, the engine + the 3 fixes are live.")
