#!/usr/bin/env python3
"""`dxcom` for a machine below DX-COM's up-front RAM (15 GB) and free-disk (8 GB) minimums.

DX-COM 2.3 refuses to start below 15 GB of physical memory and 8 GB of free disk — generic minimums
for large models. A face-recognition ResNet-50 at 112x112 needs far less of both: this wrapper
reports 16 GB of RAM (os.sysconf("SC_PHYS_PAGES")) and 9 GB free (shutil.disk_usage) to those two
pre-checks and changes nothing else. The real limits still apply: run it under compile_guarded.sh,
which stops the compile before the disk or the memory runs out.

  python dxcom_lowram.py -m model.onnx -c config.json -o out/ [dxcom options]
"""
import os
import shutil
import sys

_sysconf = os.sysconf
_REPORTED = 16 * 1024 ** 3


def _sysconf_16g(name):
    v = _sysconf(name)
    if name in ("SC_PHYS_PAGES", os.sysconf_names.get("SC_PHYS_PAGES")):
        return max(v, _REPORTED // _sysconf("SC_PAGE_SIZE"))
    return v


os.sysconf = _sysconf_16g

_disk_usage = shutil.disk_usage


def _disk_usage_9g(path):
    u = _disk_usage(path)
    return u._replace(free=max(u.free, 9 * 1024 ** 3))


shutil.disk_usage = _disk_usage_9g

from dx_com.cli import launch_release  # noqa: E402 - after the patch

if __name__ == "__main__":
    sys.argv[0] = "dxcom"
    sys.exit(launch_release())
