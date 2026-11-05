#!/usr/bin/env python3
"""InsightFace ArcFace ONNX -> an input DX-COM compiles like DeepX's own ArcFace models: RGB 0..255.

  python nhwc_input.py w600k_r50.onnx w600k_r50_in255.onnx            # NCHW [1,3,112,112] (for DX-COM)
  python nhwc_input.py w600k_r50.onnx w600k_r50_nhwc.onnx --nhwc      # NHWC [1,112,112,3]

Adds the InsightFace input normalization (x - 127.5) / 127.5 inside the graph and fixes the batch
to 1. DeepX's ArcFace models carry their normalization in the graph the same way; DX-COM's own
"normalize" preprocessing (moved into the NPU graph) produced a model whose embeddings were all
alike on this DX-COM version. The calibration loader then only converts BGR -> RGB and HWC -> CHW,
and the compiled model takes what the app feeds today: an aligned 112x112 RGB uint8 crop (NHWC).
"""
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def main(src, dst, nhwc=False, size=112):
    m = onnx.load(src)
    g = m.graph
    old = g.input[0]
    old_name = old.name
    shape = [1, size, size, 3] if nhwc else [1, 3, size, size]
    new_in = helper.make_tensor_value_info("data", TensorProto.FLOAT, shape)
    g.initializer.extend([numpy_helper.from_array(np.array(127.5, np.float32), "in_mean"),
                          numpy_helper.from_array(np.array(1.0 / 127.5, np.float32), "in_scale")])
    pre = ([helper.make_node("Transpose", ["data"], ["data_nchw"], perm=[0, 3, 1, 2], name="in_transpose")] if nhwc else [])
    pre += [helper.make_node("Sub", ["data_nchw" if nhwc else "data", "in_mean"], ["data_centered"], name="in_sub"),
            helper.make_node("Mul", ["data_centered", "in_scale"], [old_name], name="in_mul")]
    nodes = list(g.node)
    del g.node[:]
    g.node.extend(pre + nodes)
    del g.input[:]
    g.input.extend([new_in])
    for t in g.output:
        t.type.tensor_type.shape.dim[0].dim_value = 1
    onnx.checker.check_model(m)
    onnx.save(m, dst)
    print(f"{src} -> {dst}: input data {shape} RGB 0..255, output {[o.name for o in g.output]}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], nhwc="--nhwc" in sys.argv[3:])
