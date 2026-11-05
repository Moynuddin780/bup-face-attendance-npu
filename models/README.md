# Models

The compiled NPU models are **not stored in this repository** (large binaries, and the weights come with their own
licences). Put them here, or point `DET_MODEL_PATH` / `REC_MODEL_PATH` in `.env` at them.

| File | Model | How to get it |
|---|---|---|
| `scrfd-10g_640x640.dxnn` | SCRFD-10G face detector | DEEPX model zoo — downloaded by `./setup.sh` |
| `arcface_w600k_r50_112x112.dxnn` | ArcFace w600k ResNet-50 (InsightFace) | built from the ONNX model with DX-COM 2.4.0 — see [`tools/model_convert`](../tools/model_convert/README.md) |
| `arcface_iresnet100_112x112_ms1m.dxnn` | ArcFace iResNet100 (MS1M), the earlier recognition model | DEEPX model zoo — downloaded by `./setup.sh` |
