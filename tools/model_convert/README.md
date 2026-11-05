# InsightFace recognition model → DXNN (DEEPX NPU)

The scripts here turn an InsightFace ArcFace ONNX (e.g. `w600k_r50.onnx` from `buffalo_l`) into a
`.dxnn` for the DX-M1, and measure it against the model in use on the app's enrolled photos.

InsightFace's pretrained models are licensed for non-commercial research only.

## 1. DX-COM

DX-COM 2.3.0 (the compiler of this DX-AS release) in its own venv, with CPU-only torch (the default
CUDA torch is several GB):

```bash
python3 -m venv ~/model_convert/venv
~/model_convert/venv/bin/pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
~/model_convert/venv/bin/pip install --no-cache-dir \
    https://sdk.deepx.ai/release/dxcom/v2.3.0/dx_com-2.3.0-cp312-cp312-manylinux_2_31_x86_64.whl
```

DX-COM refuses to start below 15 GB RAM / 8 GB free disk. A ResNet-50 at 112×112 needs about
4.5 GB RAM and well under 1 GB of disk: `dxcom_lowram.py` reports 16 GB / 9 GB to those two
pre-checks only, and `compile_guarded.sh` stops the compile before the disk (< 1.2 GB) or memory
runs out — a full disk crashes the attendance app.

## 2. Inputs

```bash
# aligned 112x112 crops of every enrolled photo, made exactly as the engine aligns faces
.venv/bin/python tools/model_convert/crops.py --out ~/model_convert/crops
# InsightFace ONNX -> batch 1, normalization (x-127.5)/127.5 inside the graph, input "data" NCHW 0..255
~/model_convert/venv/bin/python -I tools/model_convert/nhwc_input.py w600k_r50.onnx w600k_r50_in255.onnx
```

Calibration images: ~1000 of those crops (8 per student) in one folder (`./calib`). Never ImageNet.

## 3. Compile

```bash
cd ~/model_convert
.../tools/model_convert/compile_guarded.sh venv/bin/python onnx/w600k_r50_in255.onnx w600k_r50.json out/ --gen_log
```

The compiled model takes what the app already feeds: an aligned RGB uint8 crop, NHWC `[1,112,112,3]`,
512-d output — a drop-in for the current recognition model.

## 4. Measure before using it

```bash
.venv/bin/python tools/model_convert/embed.py --crops ~/model_convert/crops --dxnn models/arcface_iresnet100_112x112_ms1m.dxnn --out old.npz
~/model_convert/venv/bin/python -I tools/model_convert/embed.py --crops ~/model_convert/crops --onnx w600k_r50.onnx --out fp32.npz
.venv/bin/python tools/model_convert/embed.py --crops ~/model_convert/crops --dxnn out/w600k_r50_in255.dxnn --out npu.npz
.venv/bin/python tools/model_convert/report.py old=old.npz w600k_fp32=fp32.npz w600k_npu=npu.npz \
    --same "Aslam=Aslam Sikder" --same "Ramim=Sakiul Mowla Ramim"
```

`report.py` prints rank-1, right / wrong / stranger acceptance per threshold, the threshold that keeps
strangers (people not enrolled) at or below 1 %, and the INT8-vs-FP32 cosine. A compiled model whose
mean cosine to its FP32 ONNX is below ~0.98 is broken or badly quantized — do not deploy it.
