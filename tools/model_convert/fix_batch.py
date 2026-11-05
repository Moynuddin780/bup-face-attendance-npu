#!/usr/bin/env python3
"""Fix an ONNX model's batch dimension to 1 (DX-COM compiles batch-1 models only).

  python fix_batch.py w600k_r50.onnx w600k_r50_b1.onnx

InsightFace exports the batch as a symbolic dimension ("None"); every input and output gets 1.
Prints the input name and the operator set, for the DX-COM JSON config.
"""
import collections
import sys

import onnx


def main(src, dst):
    m = onnx.load(src)
    for t in list(m.graph.input) + list(m.graph.output):
        dims = t.type.tensor_type.shape.dim
        if dims:
            dims[0].dim_value = 1   # clears the symbolic dim_param
    onnx.checker.check_model(m)
    onnx.save(m, dst)
    shape = lambda t: [d.dim_value or d.dim_param for d in t.type.tensor_type.shape.dim]  # noqa: E731
    print("inputs :", [(t.name, shape(t)) for t in m.graph.input])
    print("outputs:", [(t.name, shape(t)) for t in m.graph.output])
    print("opset  :", [(o.domain or "ai.onnx", o.version) for o in m.opset_import])
    print("ops    :", dict(collections.Counter(n.op_type for n in m.graph.node)))


if __name__ == "__main__":
    main(*sys.argv[1:3])
