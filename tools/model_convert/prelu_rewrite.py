#!/usr/bin/env python3
"""Replace every PRelu by Relu(x) + (-a) * Relu(-x) — the same function written without PRelu.

  python prelu_rewrite.py in.onnx out.onnx

InsightFace's arcface_torch models learned NEGATIVE PReLU slopes for almost every channel after the
first stage (w600k_r50: 250 of 256 in stage 3): a negative input gives a POSITIVE output. DX-COM 2.3.0
compiled those PRelus wrongly (the error grew block by block until the embedding carried nothing);
DeepX's own ArcFace models have positive slopes.
"""
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper


def main(src, dst):
    m = onnx.load(src)
    g = m.graph
    inits = {i.name: i for i in g.initializer}
    nodes, added, n_done = [], [], 0
    for n in g.node:
        if n.op_type != "PRelu":
            nodes.append(n)
            continue
        x, slope = n.input
        a = numpy_helper.to_array(inits[slope]).astype(np.float32)
        b = n.name or n.output[0]
        added.append(numpy_helper.from_array(-a, b + "_negslope"))
        nodes += [helper.make_node("Relu", [x], [b + "_pos"], name=b + "_relu_pos"),
                  helper.make_node("Neg", [x], [b + "_negx"], name=b + "_neg"),
                  helper.make_node("Relu", [b + "_negx"], [b + "_negpart"], name=b + "_relu_neg"),
                  helper.make_node("Mul", [b + "_negpart", b + "_negslope"], [b + "_scaled"], name=b + "_mul"),
                  helper.make_node("Add", [b + "_pos", b + "_scaled"], [n.output[0]], name=b + "_add")]
        n_done += 1
    del g.node[:]
    g.node.extend(nodes)
    g.initializer.extend(added)
    onnx.checker.check_model(m)
    onnx.save(m, dst)
    print(f"{src} -> {dst}: {n_done} PRelu rewritten")


if __name__ == "__main__":
    main(*sys.argv[1:3])
