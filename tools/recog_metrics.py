#!/usr/bin/env python3
"""Leave-one-out identification metrics over enrolled photos, using the app's own math.

Gallery per identity = l2_norm(mean(normed_embedding of its photos))   (build_embeddings_for_class)
Probe            = l2_norm(normed_embedding)                          (_process_camera)
For each probe photo its own image is left out of its identity's gallery.

  rank1        top-1 identity correct
  accept@thr   top-1 correct AND score >= SIM_THRESHOLD -> the app would mark the right student
  false@thr    top-1 WRONG  AND score >= SIM_THRESHOLD -> the app would mark the wrong student
"""
import re
import sys

import numpy as np


def identity(folder_label):
    """Folders are '<student_id>_<name>'; the same person enrolled twice shares a name."""
    name = folder_label.split("_", 1)[1] if "_" in folder_label else folder_label
    return re.sub(r"[^a-z]", "", name.lower())


def l2(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n == 0, 1e-9, n)


def evaluate(npz_path, sim_threshold=0.45, min_photos=3):
    d = np.load(npz_path, allow_pickle=False)
    embs, labels = d["embs"], [identity(x) for x in d["labels"]]
    ok = ~np.isnan(embs).any(axis=1)
    ids = sorted(set(labels))
    idx = {i: [k for k in range(len(labels)) if labels[k] == i and ok[k]] for i in ids}
    sums = {i: embs[idx[i]].sum(axis=0) if idx[i] else None for i in ids}

    probes = rank1 = accept = false_acc = 0
    genuine = []
    for i in ids:
        if len(idx[i]) < min_photos:
            continue
        for k in idx[i]:
            gal_ids, gal = [], []
            for j in ids:
                if not idx[j]:
                    continue
                if j == i:
                    g = (sums[j] - embs[k]) / (len(idx[j]) - 1)
                else:
                    g = sums[j] / len(idx[j])
                gal_ids.append(j)
                gal.append(g)
            sims = l2(np.vstack(gal)) @ l2(embs[k])
            best = int(np.argmax(sims))
            probes += 1
            correct = gal_ids[best] == i
            rank1 += correct
            accept += correct and sims[best] >= sim_threshold
            false_acc += (not correct) and sims[best] >= sim_threshold
            genuine.append(float(sims[gal_ids.index(i)]))
    return {
        "photos": int(len(labels)),
        "faces_found": int(ok.sum()),
        "detect_rate": float(ok.mean()),
        "identities": len([i for i in ids if len(idx[i]) >= min_photos]),
        "probes": probes,
        "rank1": rank1 / probes if probes else 0.0,
        f"accept@{sim_threshold}": accept / probes if probes else 0.0,
        f"false@{sim_threshold}": false_acc / probes if probes else 0.0,
        "genuine_mean": float(np.mean(genuine)) if genuine else 0.0,
        "seconds": float(d["seconds"]),
    }


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(p, evaluate(p))
