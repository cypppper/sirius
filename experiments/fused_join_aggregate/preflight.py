"""Read-only runtime prerequisites; no host or container settings are changed."""

import ctypes
import errno
import json
import os
from pathlib import Path
import resource
import subprocess

root = Path(__file__).resolve().parent
lib = ctypes.CDLL(str(Path(os.environ["CONDA_PREFIX"]) / "lib/liburing.so"))
ring = ctypes.create_string_buffer(4096)
lib.io_uring_queue_init.argtypes = [ctypes.c_uint, ctypes.c_void_p, ctypes.c_uint]
lib.io_uring_queue_init.restype = ctypes.c_int
result = lib.io_uring_queue_init(2, ctypes.byref(ring), 0)
if result == 0:
    lib.io_uring_queue_exit.argtypes = [ctypes.c_void_p]
    lib.io_uring_queue_exit(ctypes.byref(ring))
status = Path("/proc/self/status").read_text().splitlines()
info = {
    "io_uring_queue_init_result": result,
    "error": errno.errorcode.get(-result, "success" if result == 0 else "unknown"),
    "io_uring_disabled_sysctl": Path("/proc/sys/kernel/io_uring_disabled")
    .read_text()
    .strip(),
    "memlock_limit_bytes": resource.getrlimit(resource.RLIMIT_MEMLOCK),
    "process_restrictions": [
        line for line in status if line.startswith(("Seccomp:", "CapBnd:"))
    ],
    "pinned_submodules": subprocess.check_output(
        [
            "git",
            "submodule",
            "status",
            "cucascade",
            "duckdb",
            "sirius-duckdb/duckdb",
            "substrait",
        ],
        text=True,
    ).splitlines(),
    "interpretation": "io_uring setup is denied. This probe does not test Sirius initialization or the KvikIO backend.",
}
(root / "results/runtime_preflight.json").write_text(json.dumps(info, indent=2) + "\n")
print(json.dumps(info, indent=2))
