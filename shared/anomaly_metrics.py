"""No-point-adjustment anomaly-detection metrics.

Single source of truth for all model anomaly files. Given flat 1-D 0/1 numpy int
arrays ``gt`` and ``pred`` of equal length, it reports, side by side:

  RAW    - classic point-wise precision/recall/F1 (no adjustment)
  PA     - point-adjusted (Xu et al. 2018): the metric that inflates F1 by
           crediting an entire ground-truth segment for a single detected point
  EVENT  - event-wise / segment F1 (a GT event counts as detected on any overlap)
  RANGE  - range-based F1 (Tatbul et al., NeurIPS 2018; flat bias, reciprocal
           cardinality, existence weight ``alpha``)
  AFFIL  - affiliation-based F1 (Huet et al., KDD 2022) via the ``affiliation``
           package (vendored under shared/affiliation/ or pip-installed);
           reported as NaN if the package is unavailable.

The EVENT/RANGE/AFFIL metrics do NOT use point adjustment, so they reveal whether
an apparent anomaly gain survives once the point-adjustment inflation is removed.
"""

import math
import numpy as np


# ── event extraction ─────────────────────────────────────────────────────────
def events(binary):
    """Contiguous runs of 1 as [(start, end_exclusive), ...]."""
    b = np.asarray(binary).astype(int).ravel()
    if b.size == 0:
        return []
    d = np.diff(np.concatenate(([0], b, [0])))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    return list(zip(starts.tolist(), ends.tolist()))


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return float(p), float(r), float(f)


# ── point adjustment (Xu et al. 2018) ────────────────────────────────────────
def adjustment(gt, pred):
    """Point-adjust ``pred`` in place-safe copy; returns (gt, adjusted_pred)."""
    gt = np.asarray(gt).astype(int).copy()
    pred = np.asarray(pred).astype(int).copy()
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                if pred[j] == 0:
                    pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                if pred[j] == 0:
                    pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def pointwise_prf(gt, pred):
    gt = np.asarray(gt).astype(int)
    pred = np.asarray(pred).astype(int)
    tp = int(np.sum((gt == 1) & (pred == 1)))
    fp = int(np.sum((gt == 0) & (pred == 1)))
    fn = int(np.sum((gt == 1) & (pred == 0)))
    return _prf(tp, fp, fn)


# ── event-wise (segment) F1 ──────────────────────────────────────────────────
def event_wise_prf(gt, pred):
    """A GT event is a true positive if ANY predicted point overlaps it (else a
    false negative); a predicted event overlapping no GT event is a false
    positive. No point adjustment, no per-point double counting."""
    gt_ev = events(gt)
    pr_ev = events(pred)

    def ov(a, b):
        return a[0] < b[1] and b[0] < a[1]

    tp = sum(1 for g in gt_ev if any(ov(g, p) for p in pr_ev))
    fn = len(gt_ev) - tp
    fp = sum(1 for p in pr_ev if not any(ov(p, g) for g in gt_ev))
    return _prf(tp, fp, fn)


# ── range-based F1 (Tatbul et al., NeurIPS 2018) ─────────────────────────────
def range_based_prf(gt, pred, alpha=0.0):
    """Range-based precision/recall/F1 with flat positional bias and reciprocal
    cardinality. ``alpha`` weights the existence reward in recall (0 = pure
    overlap). Precision has no existence term (alpha applies to recall only)."""
    R = events(gt)
    P = events(pred)

    def inter(a, b):
        return max(0, min(a[1], b[1]) - max(a[0], b[0]))

    def length(a):
        return a[1] - a[0]

    # recall over real ranges
    if len(R) == 0:
        rec = 0.0
    else:
        tot = 0.0
        for Ri in R:
            hit = [Pj for Pj in P if inter(Ri, Pj) > 0]
            existence = 1.0 if hit else 0.0
            if hit:
                card = 1.0 if len(hit) == 1 else 1.0 / len(hit)
                overlap = card * sum(inter(Ri, Pj) / length(Ri) for Pj in hit)
            else:
                overlap = 0.0
            tot += alpha * existence + (1 - alpha) * overlap
        rec = tot / len(R)

    # precision over predicted ranges
    if len(P) == 0:
        prec = 0.0
    else:
        tot = 0.0
        for Pi in P:
            hit = [Rj for Rj in R if inter(Pi, Rj) > 0]
            if hit:
                card = 1.0 if len(hit) == 1 else 1.0 / len(hit)
                tot += card * sum(inter(Pi, Rj) / length(Pi) for Rj in hit)
        prec = tot / len(P)

    f = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return float(prec), float(rec), float(f)


# ── affiliation-based F1 (Huet et al., KDD 2022) ─────────────────────────────
_AFFIL_ERR = None
try:
    from affiliation.generics import convert_vector_to_events, infer_Trange
    from affiliation.metrics import pr_from_events
    _HAVE_AFFIL = True
except Exception as _e:            # not vendored / not installed
    _HAVE_AFFIL = False
    _AFFIL_ERR = str(_e)


def affiliation_prf(gt, pred):
    """Affiliation-based precision/recall/F1 via the reference `affiliation`
    package. Returns (nan, nan, nan) if the package isn't importable, or if
    there are no ground-truth or no predicted events (affiliation is undefined
    for an empty event set)."""
    if not _HAVE_AFFIL:
        return float("nan"), float("nan"), float("nan")
    gt = np.asarray(gt).astype(int).tolist()
    pred = np.asarray(pred).astype(int).tolist()
    try:
        gt_ev = convert_vector_to_events(gt)
        pr_ev = convert_vector_to_events(pred)
        if len(gt_ev) == 0 or len(pr_ev) == 0:
            return float("nan"), float("nan"), float("nan")
        Trange = (0, len(gt))
        res = pr_from_events(pr_ev, gt_ev, Trange)
        p = float(res["precision"])
        r = float(res["recall"])
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return p, r, f
    except Exception:
        return float("nan"), float("nan"), float("nan")


# ── aggregate ────────────────────────────────────────────────────────────────
def compute_all(gt, pred, point_adjust=True):
    """Return every metric variant in one dict.

    Primary keys ``f1/precision/recall`` follow ``point_adjust`` (True → the
    point-adjusted numbers, matching the historical default that downstream
    sweep scripts read via ``anom.get('f1')``). All variants are also returned
    under explicit suffixes so nothing is lost.
    """
    gt = np.asarray(gt).astype(int).ravel()
    pred = np.asarray(pred).astype(int).ravel()

    p_raw, r_raw, f_raw = pointwise_prf(gt, pred)
    gt_adj, pred_adj = adjustment(gt, pred)
    p_adj, r_adj, f_adj = pointwise_prf(gt_adj, pred_adj)
    p_ev, r_ev, f_ev = event_wise_prf(gt, pred)
    p_rng, r_rng, f_rng = range_based_prf(gt, pred)
    p_af, r_af, f_af = affiliation_prf(gt, pred)

    acc_raw = float(np.mean(gt == pred))
    acc_adj = float(np.mean(gt_adj == pred_adj))

    primary = dict(
        f1=f_adj, precision=p_adj, recall=r_adj, accuracy=acc_adj,
    ) if point_adjust else dict(
        f1=f_raw, precision=p_raw, recall=r_raw, accuracy=acc_raw,
    )
    return dict(
        **primary,
        point_adjust=point_adjust,
        f1_raw=f_raw, precision_raw=p_raw, recall_raw=r_raw, accuracy_raw=acc_raw,
        f1_adj=f_adj, precision_adj=p_adj, recall_adj=r_adj, accuracy_adj=acc_adj,
        f1_event=f_ev, precision_event=p_ev, recall_event=r_ev,
        f1_range=f_rng, precision_range=p_rng, recall_range=r_rng,
        f1_affiliation=f_af, precision_affiliation=p_af, recall_affiliation=r_af,
    )


def format_table(m, title="Anomaly Detection"):
    """One-block multi-column summary: RAW | PA | EVENT | RANGE | AFFIL."""
    cols = [
        ("RAW",   "raw"),
        ("PA",    "adj"),
        ("EVENT", "event"),
        ("RANGE", "range"),
        ("AFFIL", "affiliation"),
    ]

    def cell(v):
        return "  nan  " if (isinstance(v, float) and math.isnan(v)) else f"{v:7.4f}"

    lines = ["", "=" * 62, f"  [{title}]  (EVENT/RANGE/AFFIL = no point adjustment)"]
    hdr = f"  {'':10}" + "".join(f"{name:>9}" for name, _ in cols)
    lines.append(hdr)
    for label, key in (("Precision", "precision"), ("Recall", "recall"), ("F1", "f1")):
        row = f"  {label:10}"
        for _, suf in cols:
            row += f"{cell(m.get(f'{key}_{suf}', float('nan'))):>9}"
        lines.append(row)
    if not _HAVE_AFFIL:
        lines.append(f"  (affiliation unavailable: {_AFFIL_ERR}; vendor shared/affiliation/ or pip install affiliation-metrics)")
    lines.append("=" * 62)
    lines.append("")
    return "\n".join(lines)
