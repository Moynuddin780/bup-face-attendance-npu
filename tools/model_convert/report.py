#!/usr/bin/env python3
"""Compare recognition models on the same aligned crops (embed.py outputs).

  python report.py old=emb_old_npu.npz w600k_fp32=emb_w600k_fp32.npz w600k_npu=emb_w600k_npu.npz \
                   [--same "Aslam=Aslam Sikder" ...]

--same names two enrolments of one person (test accounts in two batches): counted as one student.

Per model, over every student photo as a probe (leave-one-out, the app's gallery rule — the mean
of a student's other photos, recog_metrics.py):
  rank1            the best match is the right student
  right@t / wrong@t   the app would mark the right / a WRONG student at similarity threshold t
  stranger@t       a person NOT enrolled (the probe's own student removed from the gallery) would
                   still be marked as somebody at t — the misrecognition of visitors, other sections
  t(stranger<=1%)  the lowest threshold that keeps strangers at or below 1 %, and right@ that t
  d'               separation of genuine and impostor similarities (higher = cleaner)
With an FP32 reference, the per-crop cosine between the reference and an NPU model shows what the
INT8 quantization costs.
"""
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 2)[0])
from recog_metrics import identity, l2  # noqa: E402


def analyse(npz, same=None, min_photos=3, thresholds=(0.30, 0.35, 0.40, 0.45, 0.50, 0.55)):
    d = np.load(npz, allow_pickle=False)
    ok = ~np.isnan(d["embs"]).any(axis=1)   # photos without a face (eval_recognition.py) are left out
    embs = l2(d["embs"][ok].astype(np.float64))
    same = same or {}
    labels = [same.get(identity(x), identity(x)) for x, keep in zip(d["labels"], ok) if keep]
    ids = sorted(set(labels))
    idx = {i: [k for k, l in enumerate(labels) if l == i] for i in ids}
    sums = {i: embs[idx[i]].sum(axis=0) for i in ids}
    best_right, best_wrong, best_stranger, genuine, impostor = [], [], [], [], []
    for i in ids:
        if len(idx[i]) < min_photos:
            continue
        others = [j for j in ids if j != i]
        gal_o = l2(np.vstack([sums[j] / len(idx[j]) for j in others]))
        for k in idx[i]:
            own = l2((sums[i] - embs[k]) / (len(idx[i]) - 1))
            so = gal_o @ embs[k]
            s_own = float(own @ embs[k])
            top_o = float(so.max())
            genuine.append(s_own)
            impostor.append(top_o)
            best_right.append(s_own if s_own >= top_o else -1.0)   # the right student wins, at this score
            best_wrong.append(top_o if top_o > s_own else -1.0)    # a wrong student wins, at this score
            best_stranger.append(top_o)                            # not enrolled: the best other match
    br, bw, bs = map(np.array, (best_right, best_wrong, best_stranger))
    g, im = np.array(genuine), np.array(impostor)
    out = {"probes": len(br), "rank1": float((br >= 0).mean()),
           "d'": float((g.mean() - im.mean()) / np.sqrt((g.var() + im.var()) / 2)),
           "genuine_mean": float(g.mean()), "impostor_top_mean": float(im.mean())}
    for t in thresholds:
        out[f"right@{t}"] = float((br >= t).mean())
        out[f"wrong@{t}"] = float((bw >= t).mean())
        out[f"stranger@{t}"] = float((bs >= t).mean())
    grid = np.arange(0.20, 0.80, 0.005)
    ok = [t for t in grid if (bs >= t).mean() <= 0.01]
    t1 = float(ok[0]) if ok else float("nan")
    out["t(stranger<=1%)"] = round(t1, 3)
    out["right@t1"] = float((br >= t1).mean()) if ok else 0.0
    out["wrong@t1"] = float((bw >= t1).mean()) if ok else 0.0
    return out, embs


def main(args):
    same, pairs = {}, []
    it = iter(args)
    for a in it:
        if a == "--same":
            x, y = next(it).split("=", 1)
            same[identity("_" + x)] = identity("_" + y)
        else:
            pairs.append(a)
    models = dict(p.split("=", 1) for p in pairs)
    res, embs = {}, {}
    for name, path in models.items():
        res[name], embs[name] = analyse(path, same)
    keys = list(next(iter(res.values())).keys())
    w = max(len(k) for k in keys) + 2
    print("".ljust(w) + "".join(n.rjust(14) for n in res))
    for k in keys:
        print(k.ljust(w) + "".join((f"{res[n][k]:.3f}" if isinstance(res[n][k], float) else str(res[n][k])).rjust(14)
                                   for n in res))
    refs = [n for n in res if n.endswith("fp32")]
    for ref in refs:
        base = ref[:-len("fp32")]
        for n in res:
            if n != ref and n.startswith(base) and embs[n].shape == embs[ref].shape:
                cos = (embs[n] * embs[ref]).sum(axis=1)
                print(f"\nINT8 {n} vs {ref}: cosine mean {cos.mean():.4f}, min {cos.min():.4f}, "
                      f"p1 {np.percentile(cos, 1):.4f}, <0.95: {(cos < 0.95).mean() * 100:.1f}%")


if __name__ == "__main__":
    main(sys.argv[1:])
