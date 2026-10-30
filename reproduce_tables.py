"""Recompute the main numbers of the paper from the released per-item files.

Table 1: per-tool document alteration rates (Wilson 95% CI), altered/total units by group, and evasion.
H2a:     Spearman correlation between tool-level unit retention and evasion.
Table 2: AUROC (human originals vs AI drafts / vs 12B rewrites) with article-level bootstrap 95% CIs,
         and TPR at a 5% false-positive threshold set on the human originals.

Usage: python reproduce_tables.py [--data data]
Only numpy is required.
"""
import argparse, collections, csv, json, math, os

import numpy as np

FILES = ["bf16", "Q8_0", "Q3-QAT", "IQ2_XS-QAT"]
DETECTORS = [("Jev (decision model)", "jev_p_ai", 1), ("Fast-DetectGPT", "fast_detectgpt_crit", 1),
             ("Binoculars", "binoculars", -1), ("RoBERTa detector", "roberta_p_fake", 1), ("RADAR", "radar_p_ai", 1)]
B, SEED = 2000, 20261025


def rankdata(x):
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(1, len(x) + 1)
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = (i + j + 2) / 2
        i = j + 1
    return r


def auroc(pos, neg):
    """Probability that a positive (AI) text scores higher than a negative (human) one; ties count half."""
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def spearman(a, b):
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def wilson(k, n, z=1.96):
    p, d = k / n, 1 + z * z / n
    c, h = p + z * z / (2 * n), z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def table1(docs):
    print("Table 1. Fact alteration (altered/total).")
    print(f"{'tool':<18}{'altered docs':>13}  {'rate [95% CI]':<20}{'numbers':>9}{'names':>9}{'quoted':>8}{'evasion':>9}")
    groups = collections.OrderedDict()
    for d in docs:
        groups.setdefault(d["tool"] if d["set"] == "commercial" else "12B " + d["model_file"], []).append(d)
    commercial = [g for g in groups if not g.startswith("12B ")]
    commercial.sort(key=lambda g: (-sum(any(not u["kept"] for u in d["units"]) for d in groups[g] if d["units"]), g))
    retention, evasion = [], []
    for g in commercial + ["12B " + f for f in FILES]:
        ds = groups[g]
        eligible = [d for d in ds if d["units"]]
        k = sum(any(not u["kept"] for u in d["units"]) for d in eligible)
        lo, hi = wilson(k, len(eligible))
        cells = []
        for grp in ("num", "ent", "quote"):
            us = [u for d in ds for u in d["units"] if u["group"] == grp]
            cells.append(f"{sum(not u['kept'] for u in us)}/{len(us)}")
        ev = "--"
        if g in commercial:
            units = [u for d in ds for u in d["units"]]
            retention.append(sum(u["kept"] for u in units) / len(units))
            evasion.append(float(np.mean([d["aihb_mean_human_prob"] for d in ds])))
            ev = f"{evasion[-1]:.2f}"
        print(f"{g:<18}{f'{k}/{len(eligible)}':>13}  {f'{k / len(eligible):.2f} [{lo:.2f}, {hi:.2f}]':<20}"
              f"{cells[0]:>9}{cells[1]:>9}{cells[2]:>8}{ev:>9}")
    print(f"\nH2a. Spearman rho between unit retention and evasion across {len(commercial)} tools: "
          f"{spearman(retention, evasion):.2f}\n")


def table2(passages, scores):
    pids = [p["passage_id"] for p in passages]
    article = {p["passage_id"]: p["article_url"] for p in passages}
    arts, members = [], {}
    for p in pids:
        if article[p] not in members:
            arts.append(article[p])
            members[article[p]] = []
        members[article[p]].append(p)

    def get(ps, text_type, col, sign):
        return np.array([sign * scores[(p, text_type)][col] for p in ps])

    def aucs(ps, col, sign):
        h = get(ps, "human", col, sign)
        d = get(ps, "ai_draft", col, sign)
        rw = np.concatenate([get(ps, "rewrite_" + f, col, sign) for f in FILES])
        return auroc(d, h), auroc(rw, h), h, d, rw

    print(f"Table 2. Detection on {len(pids)} federal passages ({len(arts)} articles); "
          f"CIs resample articles ({B} draws, seed {SEED}).")
    print(f"{'detector':<22}{'AUROC draft [95% CI]':<26}{'AUROC rewrite [95% CI]':<26}{'TPR draft':>10}{'TPR rewrite':>12}")
    for name, col, sign in DETECTORS:
        a_d, a_r, h, d, rw = aucs(pids, col, sign)
        rng = np.random.default_rng(SEED)
        boot_d, boot_r = [], []
        for _ in range(B):
            pick = rng.choice(len(arts), len(arts), replace=True)
            ps = [p for i in pick for p in members[arts[i]]]
            bd, br, *_ = aucs(ps, col, sign)
            boot_d.append(bd)
            boot_r.append(br)
        t = np.percentile(h, 95)
        ci = lambda xs: f"[{np.percentile(xs, 2.5):.3f}, {np.percentile(xs, 97.5):.3f}]"
        print(f"{name:<22}{f'{a_d:.3f} {ci(boot_d)}':<26}{f'{a_r:.3f} {ci(boot_r)}':<26}"
              f"{np.mean(d > t):>10.3f}{np.mean(rw > t):>12.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    a = ap.parse_args()
    docs = [json.loads(l) for l in open(os.path.join(a.data, "fact_units.jsonl"), encoding="utf-8") if l.strip()]
    passages = list(csv.DictReader(open(os.path.join(a.data, "passages.csv"), encoding="utf-8")))
    scores = {}
    for r in csv.DictReader(open(os.path.join(a.data, "detection_scores.csv"), encoding="utf-8")):
        scores[(r["passage_id"], r["text_type"])] = {k: float(v) for k, v in r.items() if k not in ("passage_id", "text_type")}
    table1(docs)
    table2(passages, scores)


if __name__ == "__main__":
    main()
