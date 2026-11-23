#!/usr/bin/env python3
"""
analyze_layer_exit.py — tables for the layer-wise probing / early-exit study
(scripts/run_layer_exit.sh).

For every (model, seed, dataset, horizon) and every exit k it reads the probe's
best validation loss and its test MSE from the log. Validation and test are
paired by position: each model prints its validation losses for a horizon
before that horizon's test MSE, so the val values in the log segment ending at
the i-th test MSE belong to horizon i (the best = minimum = the restored head).

Outputs
  1. Test MSE (mean over horizons) per exit k, mean ± std over seeds.
  2. Is an intermediate layer better than the final one (k=24)?
  3. Early exit: per (model, seed, dataset, horizon) pick k* = argmin val loss,
     report its test MSE and k* against the full encoder (k=24).

Usage:
  python scripts/analyze_layer_exit.py [--logs logs/layer_exit] [--csv out.csv]
"""
import argparse
import csv
import re
import statistics as st
from pathlib import Path

HORIZONS = [96, 192, 336, 720]
MODELS = ["dino", "jepa", "lejepa", "mae", "ntp", "softclt", "diffusion"]

TEST = {
    "dino":      r"Mean MSE:\s+([\d.]+)",
    "softclt":   r"Mean MSE:\s+([\d.]+)",
    "jepa":      r"\[TRAINED\] MSE\s+—\s+P2P:\s+([\d.]+)",
    "lejepa":    r"^\s+MSE: ([\d.]+)\s+MAE",
    "mae":       r"\[MAE\] pred_len=\d+\s+MSE=([\d.]+)",
    "ntp":       r"NTP \(pretrained\)\s+([\d.]+)",
    "diffusion": r"pred_len=\d+ → test MSE=([\d.]+)",
}
VAL = {
    "dino":      r"\[DINO forecast\] epoch \d+ — val MSE: ([\d.]+)",
    "softclt":   r"\[DINO forecast\] epoch \d+ — val MSE: ([\d.]+)",
    "jepa":      r"\[TRAINED\] Restored best checkpoint\s+\(best val MSE=([\d.]+)\)",
    "lejepa":    r"\[Forecast\] Restored best checkpoint\s+\(best val MSE=([\d.]+)\)",
    "mae":       r"Better model found at epoch \d+ with valid_loss value: (\d+\.\d+)",
    "ntp":       r"Best val loss: ([\d.]+)",
    "diffusion": r"Vali Loss: ([\d.]+)",
}
LOGNAME = re.compile(r"(?P<m>[a-z]+)_seed(?P<s>\d+)_exit(?P<k>\d+)_(?P<d>\w+)\.log$")


def parse(path: Path, model: str):
    """[(val, test)] per horizon, or None if the run is incomplete."""
    t = path.read_text(errors="ignore").replace("\r", "\n")
    tests = list(re.finditer(TEST[model], t, re.M))
    if len(tests) < len(HORIZONS):
        return None
    tests = tests[-len(HORIZONS):]
    vals = [(m.start(), float(m.group(1))) for m in re.finditer(VAL[model], t, re.M)]
    out, prev = [], 0
    for i, tm in enumerate(tests):
        seg = [v for pos, v in vals if prev <= pos < tm.start()]
        out.append((min(seg) if seg else None, float(tm.group(1))))
        prev = tm.end()
    return out


def ms(v):
    v = [x for x in v if x is not None]
    if not v:
        return "   --         "
    return f"{st.mean(v):.4f} ± {st.stdev(v):.4f}" if len(v) > 1 else f"{v[0]:.4f} (n=1)  "


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="logs/layer_exit")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--full", default="logs/depth_24layers",
                    help="k=24 (full encoder) is read from here when there is no exit24 log: "
                         "the depth-sweep 24-layer forecasting run is the same backbone and protocol")
    a = ap.parse_args()

    R = {}      # (m, s, d, k) -> [(val, test)] * 4
    warn = []
    for p in sorted(Path(a.logs).glob("*_forecast/*.log")):
        g = LOGNAME.search(p.name)
        if not g or g["m"] not in TEST:
            continue
        m, s, k, d = g["m"], int(g["s"]), int(g["k"]), g["d"]
        txt = p.read_text(errors="ignore")
        if k < 24 and "[exit] encoder truncated" not in txt:
            warn.append(f"{p.name}: no '[exit] encoder truncated' line — truncation not applied?")
        r = parse(p, m)
        if r is None:
            warn.append(f"{p.name}: incomplete"); continue
        if any(v is None for v, _ in r):
            warn.append(f"{p.name}: validation loss missing for some horizon")
        R[(m, s, d, k)] = r

    # k=24 = the untruncated encoder: reuse the depth-sweep 24-layer run
    for (m, s, d) in sorted({(m, s, d) for m, s, d, _ in R}):
        if (m, s, d, 24) in R:
            continue
        p = Path(a.full) / f"{m}_forecast" / f"{m}_seed{s}_{d}.log"
        r = parse(p, m) if p.exists() else None
        if r is None:
            warn.append(f"{p}: missing/incomplete — no k=24 for {m} seed {s} {d}")
        else:
            R[(m, s, d, 24)] = r

    ks = sorted({k for *_, k in R})
    ds = sorted({d for _, _, d, _ in R})
    seeds = sorted({s for _, s, _, _ in R})
    models = [m for m in MODELS if any(key[0] == m for key in R)]

    def avg_test(m, s, d, k):
        r = R.get((m, s, d, k))
        return st.mean(t for _, t in r) if r else None

    # 1. layer-wise test MSE
    print("1. Test MSE (mean over horizons 96/192/336/720), mean ± std over seeds", seeds)
    for d in ds:
        print(f"\n  {d}" + "".join(f"{'k='+str(k):>18s}" for k in ks) + "     best k")
        for m in models:
            means = {k: [avg_test(m, s, d, k) for s in seeds] for k in ks}
            mu = {k: st.mean([x for x in v if x is not None]) for k, v in means.items()
                  if any(x is not None for x in v)}
            best = min(mu, key=mu.get) if mu else None
            print(f"  {m:9s}" + "".join(f"{ms(means[k]):>18s}" for k in ks)
                  + (f"     {best}" if best else ""))

    # 2. intermediate vs final
    print("\n2. Best intermediate layer vs final layer (k=24), test MSE averaged over seeds")
    for d in ds:
        for m in models:
            mu = {}
            for k in ks:
                v = [x for x in (avg_test(m, s, d, k) for s in seeds) if x is not None]
                if v:
                    mu[k] = st.mean(v)
            if 24 not in mu or len(mu) < 2:
                continue
            inter = {k: v for k, v in mu.items() if k != 24}
            kb = min(inter, key=inter.get)
            delta = inter[kb] - mu[24]
            tag = "intermediate BETTER" if delta < 0 else "final better"
            print(f"  {d:6s} {m:9s} best k={kb:2d} {inter[kb]:.4f}  vs k=24 {mu[24]:.4f}"
                  f"   Δ={delta:+.4f} ({100*delta/mu[24]:+.1f}%)  {tag}")

    # 3. early exit
    print("\n3. Early exit: k* = argmin validation loss, per (seed, dataset, horizon)")
    print("   test MSE at k* vs full encoder, and blocks executed (mean over seeds & horizons)")
    rows = []
    for d in ds:
        for m in models:
            ee, full, kstar, oracle = [], [], [], []
            for s in seeds:
                if not all((m, s, d, k) in R for k in ks):
                    continue
                for h in range(len(HORIZONS)):
                    vt = {k: R[(m, s, d, k)][h] for k in ks}
                    if any(v is None for v, _ in vt.values()):
                        continue
                    k_sel = min(ks, key=lambda k: vt[k][0])
                    ee.append(vt[k_sel][1]); full.append(vt[24][1]); kstar.append(k_sel)
                    oracle.append(min(t for _, t in vt.values()))
                    rows.append(dict(model=m, seed=s, dataset=d, horizon=HORIZONS[h], k_star=k_sel,
                                     test_exit=vt[k_sel][1], test_full=vt[24][1],
                                     **{f"val_k{k}": vt[k][0] for k in ks},
                                     **{f"test_k{k}": vt[k][1] for k in ks}))
            if not ee:
                continue
            e, f = st.mean(ee), st.mean(full)
            print(f"  {d:6s} {m:9s} early-exit {e:.4f}  full {f:.4f}  Δ={e-f:+.4f} ({100*(e-f)/f:+.1f}%)"
                  f"   oracle {st.mean(oracle):.4f}   blocks {st.mean(kstar):4.1f}/24"
                  f" ({100*st.mean(kstar)/24:.0f}%)   k*={kstar}")

    if a.csv and rows:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        print(f"\nper-horizon rows → {a.csv}")
    if warn:
        print("\nWARNINGS"); [print("  " + x) for x in warn]


if __name__ == "__main__":
    main()
