#!/usr/bin/env python3
"""Replace every BatchNormalization of an ONNX model by the same per-channel affine as a 1x1
depthwise Conv (mode dwconv) or Mul + Add (mode muladd). Mathematically identical (checked with
onnxruntime); it only changes how DX-COM sees the operation.

  python bn_rewrite.py in.onnx out.onnx [dwconv|muladd]

Why: the InsightFace IResNet applies a BatchNormalization to the residual sum at the start of each
block, while the un-normalized sum also feeds the block's shortcut. DX-COM 2.3.0 compiled such a
model into an NPU model whose embeddings were all alike (layer-by-layer comparison in
tools/model_convert/README.md); DeepX's own ArcFace models do not have that pattern.
"""
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper


def main(src, dst, mode="dwconv"):
    m = onnx.shape_inference.infer_shapes(onnx.load(src))
    g = m.graph
    rank = {v.name: len(v.type.tensor_type.shape.dim) for v in list(g.value_info) + list(g.input) + list(g.output)}
    inits = {i.name: i for i in g.initializer}
    arr = lambda n: numpy_helper.to_array(inits[n]).astype(np.float64)  # noqa: E731
    nodes, added = [], []
    for n in g.node:
        if n.op_type != "BatchNormalization":
            nodes.append(n)
            continue
        eps = next((a.f for a in n.attribute if a.name == "epsilon"), 1e-5)
        x, gamma, beta, mean, var = n.input[:5]
        scale = arr(gamma) / np.sqrt(arr(var) + eps)
        shift = arr(beta) - arr(mean) * scale
        c = scale.size
        base = n.name or n.output[0]
        if mode == "muladd" or rank.get(x, 4) == 2:
            # [C,1,1] broadcasts over NCHW; a 2-D input (the final feature BN) keeps [C]
            shape = [c] if rank.get(x, 4) == 2 else [c, 1, 1]
            added += [numpy_helper.from_array(scale.reshape(shape).astype(np.float32), base + "_s"),
                      numpy_helper.from_array(shift.reshape(shape).astype(np.float32), base + "_b")]
            nodes += [helper.make_node("Mul", [x, base + "_s"], [base + "_m"], name=base + "_mul"),
                      helper.make_node("Add", [base + "_m", base + "_b"], [n.output[0]], name=base + "_add")]
        else:
            added += [numpy_helper.from_array(scale.reshape(c, 1, 1, 1).astype(np.float32), base + "_w"),
                      numpy_helper.from_array(shift.astype(np.float32), base + "_b")]
            nodes.append(helper.make_node("Conv", [x, base + "_w", base + "_b"], [n.output[0]], name=base + "_dw",
                                          group=c, kernel_shape=[1, 1]))
    del g.node[:]
    g.node.extend(nodes)
    g.initializer.extend(added)
    del g.value_info[:]
    onnx.checker.check_model(m)
    onnx.save(m, dst)
    print(f"{src} -> {dst}: {sum(1 for n in nodes if n.name.endswith(('_dw', '_mul')))} BatchNormalization replaced ({mode})")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "dwconv")
