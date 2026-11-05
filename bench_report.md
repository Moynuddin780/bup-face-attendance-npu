| Mode | Stream decoded -> processed | Processed fps / camera | CPU cores (process) | Students seen |
|---|---|---|---|---|
| original | 1920x1080 -> 1280x720 (squashed if not 16:9) | 8.7 (1.0x) | 1.29 | 6 |
| cpp-main | 1920x1080 -> 1280x720 | 14.5 (1.7x) | 0.72 | 6 |
| cpp-sub | 704x576 -> 1024x576 | 14.4 (1.7x) | 0.42 | 6 |

> NPU shared with `sdvision-edge` during this run (NPU_ALL refused, per-core bindings) — absolute numbers are lower than on a dedicated NPU.
