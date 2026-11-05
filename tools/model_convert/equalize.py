#!/usr/bin/env python3
"""Per-channel equalization of an IResNet's residual streams, so that it survives INT8.

  python equalize.py in_dwconv.onnx out.onnx --crops ~/model_convert/crops [--target median]

The residual stream of an ArcFace IResNet (the sum carried from block to block inside a stage)
has a few channels 15-20x larger than the rest (w600k_r50: one channel reaches 14.5 where 99.9 %
of the values are below 0.84). INT8 has one scale per tensor: it either clips those channels or
loses the others — DX-COM's model of w600k_r50 produced embeddings that were all alike.

Each channel c of a stage's stream is scaled by k_c = target / max_c (max over the crops):
  - the convolutions that write into the stream (each block's last conv, the stage's downsample
    conv) get output channel c multiplied by k_c;
  - everything that reads the stream (the next block's input normalization, the next stage's
    downsample conv and input normalization, the final bn2) gets input channel c divided by k_c.
The function of the network is unchanged (checked with onnxruntime at the end). Run bn_rewrite.py
first (muladd: DX-COM 2.3.0 compiled the dwconv form into a model that hung the NPU): the
normalizations are then per-channel Mul + Add, whose input scaling is exact.
"""
import argparse
import os
from collections import defaultdict

import numpy as np
import onnx
from onnx import numpy_helper


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--crops", required=True)
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--kmin", type=float, default=1 / 64)
    ap.add_argument("--kmax", type=float, default=8.0)
    args = ap.parse_args()

    import cv2
    import onnxruntime as ort

    m = onnx.load(args.src)
    g = m.graph
    producer = {o: n for n in g.node for o in n.output}
    consumers = defaultdict(list)
    for n in g.node:
        for i in n.input:
            consumers[i].append(n)
    inits = {i.name: i for i in g.initializer}

    # residual streams: Add outputs, grouped into stages through the identity shortcuts
    # residual Adds only: both inputs computed (a conv or the stream), not a constant shift
    adds = [n for n in g.node if n.op_type == "Add"
            and all(i not in inits and i in producer and producer[i].op_type in ("Conv", "Add") for i in n.input)]
    add_out = {n.output[0]: n for n in adds}
    parent = {t: t for t in add_out}

    def find(t):
        while parent[t] != t:
            parent[t] = parent[parent[t]]
            t = parent[t]
        return t
    for n in adds:
        for i in n.input:
            if i in add_out:
                parent[find(i)] = find(n.output[0])
    stages = defaultdict(list)
    for t in add_out:
        stages[find(t)].append(t)
    stages = [sorted(v, key=lambda t: [x.name for x in g.node].index(add_out[t].name)) for v in stages.values()]

    # per-channel maxima of every stream tensor over the crops
    tapped = onnx.load(args.src)
    for st in stages:
        for t in st:
            tapped.graph.output.append(onnx.helper.make_tensor_value_info(t, onnx.TensorProto.FLOAT, None))
    sess = ort.InferenceSession(tapped.SerializeToString(), providers=["CPUExecutionProvider"])
    names = [o.name for o in sess.get_outputs()]
    in_name = sess.get_inputs()[0].name
    nchw = sess.get_inputs()[0].shape[1] == 3
    idx = np.load(os.path.join(args.crops, "index.npz"))
    step = max(1, len(idx["files"]) // args.samples)
    chmax = {}
    for rel in idx["files"][::step]:
        rgb = cv2.cvtColor(cv2.imread(os.path.join(args.crops, str(rel))), cv2.COLOR_BGR2RGB).astype(np.float32)
        x = rgb.transpose(2, 0, 1)[None] if nchw else rgb[None]
        for name, v in zip(names, sess.run(None, {in_name: x})):
            if name in add_out:
                cm = np.abs(v[0]).reshape(v.shape[1], -1).max(axis=1)
                chmax[name] = np.maximum(chmax.get(name, 0), cm)

    def w(name):
        return numpy_helper.to_array(inits[name]).astype(np.float64)

    def set_w(name, arr):
        inits[name].CopyFrom(numpy_helper.from_array(arr.astype(np.float32), name))

    for st in stages:
        mx = np.max(np.stack([chmax[t] for t in st]), axis=0)
        target = float(np.median(mx))
        k = np.clip(target / np.maximum(mx, 1e-6), args.kmin, args.kmax)
        stream = set(st)
        # writers: the inputs of the stage's Adds that are not the stream itself
        for t in st:
            for i in add_out[t].input:
                if i in stream:
                    continue
                p = producer[i]
                assert p.op_type == "Conv", f"stream writer {p.name} is {p.op_type}"
                wt = w(p.input[1])
                set_w(p.input[1], wt * k.reshape(-1, 1, 1, 1))
                if len(p.input) > 2:
                    set_w(p.input[2], w(p.input[2]) * k)
                else:
                    raise SystemExit(f"{p.name} has no bias")
        # readers: everything outside the stage's Adds that consumes a stream tensor
        for t in st:
            for c in consumers[t]:
                if c.op_type == "Add" and c.output[0] in stream:
                    continue
                if c.op_type == "Mul":   # a normalization rewritten as Mul + Add (bn_rewrite.py muladd)
                    sname = next(i for i in c.input if i in inits)
                    sc = w(sname)
                    set_w(sname, sc / k.reshape(sc.shape))
                    continue
                assert c.op_type == "Conv", f"stream reader {c.name} is {c.op_type}"
                group = next((a.i for a in c.attribute if a.name == "group"), 1)
                wt = w(c.input[1])
                if group == wt.shape[0] and wt.shape[1] == 1:   # depthwise (a rewritten normalization)
                    set_w(c.input[1], wt / k.reshape(-1, 1, 1, 1))
                else:
                    set_w(c.input[1], wt / k.reshape(1, -1, 1, 1))
        print(f"stage of {len(st)} blocks, {len(mx)} channels: max {mx.max():.2f} -> {float((mx * k).max()):.2f}, "
              f"median {target:.2f}")

    onnx.checker.check_model(m)
    onnx.save(m, args.dst)

    # unchanged function
    a = ort.InferenceSession(args.src, providers=["CPUExecutionProvider"])
    b = ort.InferenceSession(args.dst, providers=["CPUExecutionProvider"])
    diffs = []
    for rel in idx["files"][::max(1, len(idx["files"]) // 20)]:
        rgb = cv2.cvtColor(cv2.imread(os.path.join(args.crops, str(rel))), cv2.COLOR_BGR2RGB).astype(np.float32)
        x = rgb.transpose(2, 0, 1)[None] if nchw else rgb[None]
        ea, eb = a.run(None, {in_name: x})[0].ravel(), b.run(None, {in_name: x})[0].ravel()
        diffs.append(float(ea @ eb / np.linalg.norm(ea) / np.linalg.norm(eb)))
    print(f"{args.dst}: embedding cosine to the original {min(diffs):.6f} (min over {len(diffs)} crops)")


if __name__ == "__main__":
    main()
