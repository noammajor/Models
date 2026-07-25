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
    """Point-adjust ``pred``; returns (gt, adjusted_pred).

    Vectorised equivalent of the Xu et al. (2018) rule: any ground-truth event
    that contains at least one detected point is credited as fully detected.
    O(sum of GT-event lengths) via numpy slicing instead of a Python loop over
    every timestep (which is intractable for large flattened arrays, e.g. SWaT).
    """
    gt = np.asarray(gt).astype(int).ravel()
    pred = np.asarray(pred).astype(int).ravel().copy()
    for s, e in events(gt):
        if pred[s:e].any():
            pred[s:e] = 1
    return gt, pred


def pointwise_prf(gt, pred):
    gt = np.asarray(gt).astype(int)
    pred = np.asarray(pred).astype(int)
    tp = int(np.sum((gt == 1) & (pred == 1)))
    fp = int(np.sum((gt == 0) & (pred == 1)))
    fn = int(np.sum((gt == 1) & (pred == 0)))
    return _prf(tp, fp, fn)


# ── event-wise (segment) F1 ──────────────────────────────────────────────────
def _starts_ends(ev):
    """Sorted start/end arrays from an event list (events() is already sorted)."""
    if not ev:
        return np.empty(0, int), np.empty(0, int)
    s = np.fromiter((a[0] for a in ev), int, len(ev))
    e = np.fromiter((a[1] for a in ev), int, len(ev))
    return s, e


def _overlap_exists(A_s, A_e, b_s, b_e):
    """For each query interval [b_s[i], b_e[i]), does any A event overlap it?
    A events are sorted & non-overlapping → overlap iff some A has end > b_s and
    start < b_e, a contiguous slice found by searchsorted. Fully vectorised."""
    if A_s.size == 0:
        return np.zeros(len(b_s), dtype=bool)
    lo = np.searchsorted(A_e, b_s, side="right")   # first A with end > b_s
    hi = np.searchsorted(A_s, b_e, side="left")    # first A with start >= b_e
    return lo < hi


def event_wise_prf(gt, pred):
    """A GT event is a true positive if ANY predicted point overlaps it (else a
    false negative); a predicted event overlapping no GT event is a false
    positive. No point adjustment, no per-point double counting.

    Vectorised via searchsorted interval sweeps — O((|R|+|P|)·log) instead of the
    O(|R|·|P|) nested scan, which is intractable when predictions fragment into
    hundreds of thousands of events (SWaT)."""
    gt_ev = events(gt)
    pr_ev = events(pred)
    G_s, G_e = _starts_ends(gt_ev)
    P_s, P_e = _starts_ends(pr_ev)

    tp = int(np.count_nonzero(_overlap_exists(P_s, P_e, G_s, G_e))) if gt_ev else 0
    fn = len(gt_ev) - tp
    fp = int(np.count_nonzero(~_overlap_exists(G_s, G_e, P_s, P_e))) if pr_ev else 0
    return _prf(tp, fp, fn)


# ── range-based F1 (Tatbul et al., NeurIPS 2018) ─────────────────────────────
def range_based_prf(gt, pred, alpha=0.0):
    """Range-based precision/recall/F1 with flat positional bias and reciprocal
    cardinality. ``alpha`` weights the existence reward in recall (0 = pure
    overlap). Precision has no existence term (alpha applies to recall only)."""
    R = events(gt)
    P = events(pred)
    R_s, R_e = _starts_ends(R)
    P_s, P_e = _starts_ends(P)

    def _score(Q_s, Q_e, A_s, A_e, with_existence):
        """Mean over query ranges Q of reciprocal-cardinality overlap with A
        ranges (plus optional existence reward). searchsorted finds the
        contiguous A slice overlapping each Q; intersections are vectorised."""
        if Q_s.size == 0:
            return 0.0
        tot = 0.0
        for qs, qe in zip(Q_s.tolist(), Q_e.tolist()):
            Ln = qe - qs
            if A_s.size:
                lo = np.searchsorted(A_e, qs, side="right")
                hi = np.searchsorted(A_s, qe, side="left")
            else:
                lo = hi = 0
            k = hi - lo
            if k > 0:
                inter = np.minimum(qe, A_e[lo:hi]) - np.maximum(qs, A_s[lo:hi])
                inter = inter[inter > 0]
                card = 1.0 if k == 1 else 1.0 / k
                overlap = card * float(inter.sum()) / Ln
                existence = 1.0
            else:
                overlap = 0.0
                existence = 0.0
            tot += (alpha * existence + (1 - alpha) * overlap) if with_existence else overlap
        return tot / len(Q_s)

    rec  = _score(R_s, R_e, P_s, P_e, with_existence=True)    # existence reward on recall only
    prec = _score(P_s, P_e, R_s, R_e, with_existence=False)

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


# The vendored affiliation_partition is O(|gt_ev| * |pred_ev|) pure Python, so it
# explodes when predictions fragment into hundreds of thousands of events on the
# stride-1 flattened arrays. Cap the work so compute_all can never hang; above the
# cap affiliation is reported as NaN (intractable for the reference implementation).
_AFFIL_MAX_WORK = 10_000_000


def affiliation_prf(gt, pred):
    """Affiliation-based precision/recall/F1 via the reference `affiliation`
    package. Returns (nan, nan, nan) if the package isn't importable, if there are
    no ground-truth / no predicted events, or if the event counts make the
    reference implementation intractable (see _AFFIL_MAX_WORK)."""
    if not _HAVE_AFFIL:
        return float("nan"), float("nan"), float("nan")
    # events() yields [start, end_exclusive) int tuples — exactly the affiliation
    # convention (a 1 at index i → interval [i, i+1)). Build from numpy (fast)
    # instead of a 48M-element .tolist()+convert_vector_to_events scan every call.
    gt_ev = [(int(s), int(e)) for s, e in events(gt)]
    pr_ev = [(int(s), int(e)) for s, e in events(pred)]
    if len(gt_ev) == 0 or len(pr_ev) == 0:
        return float("nan"), float("nan"), float("nan")
    if len(gt_ev) * len(pr_ev) > _AFFIL_MAX_WORK:
        return float("nan"), float("nan"), float("nan")
    try:
        Trange = (0, int(np.asarray(gt).size))
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
