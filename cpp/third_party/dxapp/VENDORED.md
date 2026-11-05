# Vendored dx_app components

Copied unchanged from `dx-all-suite/dx-runtime/dx_app/src/cpp_example/common/` at commit
`c6c35e46ae1c0ef1542a53a613eeb36a24a4dbc8` (2026-08-03). They are licensed by DEEPX to DEEPX
NPU customers (see `LICENSE`).

| File | Used for |
|---|---|
| `common/base/i_processor.hpp` | `IPreprocessor` / `IPostprocessor` interfaces, result types |
| `common/base/i_factory.hpp`, `common/base/i_visualizer.hpp` | `IFaceDetectionFactory`, `IEmbeddingFactory`, `IVisualizer` |
| `common/utility/preprocessing.hpp` | `makeLetterboxImage` (aspect-preserving letterbox) |
| `common/processors/letterbox_preprocessor.hpp` | SCRFD preprocessor (`DetectionPreprocessor`) |
| `common/processors/simple_resize_preprocessor.hpp` | ArcFace preprocessor |
| `common/processors/embedding_postprocessor.hpp` | `GenericEmbeddingPostprocessor` (L2-normalized embedding) |
| `common/processors/scrfd_face_postprocessor.hpp`, `postprocess_utils.hpp` | SCRFD anchor decode + NMS (`SCRFDPostProcess`) |

The model-specific wiring lives in `cpp/include/bup/factories.hpp`, which follows the same pattern
as dx_app's `scrfd10g` and `arcface_iresnet100_ms1m` examples.

## Not part of the Git repository

These files belong to DEEPX and their licence (`LICENSE`) forbids passing them on, so `.gitignore` leaves them out
and only this note is committed. After cloning, restore them from your own DEEPX runtime installation:

```bash
DX_APP=/path/to/dx-runtime/dx_app           # your licensed copy
cp -r "$DX_APP/src/cpp_example/common" cpp/third_party/dxapp/common
cp "$DX_APP/LICENSE" cpp/third_party/dxapp/LICENSE 2>/dev/null || true
```
