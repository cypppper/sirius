"""Extract portable memory and Nsight diagnostics without rerunning GPU timings."""

import csv
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parent / "results/sirius"


def main():
    memory = []
    for folder in sorted(ROOT.glob("e2e_t*_p*_*_sf*")):
        match = re.fullmatch(
            r"e2e_t(\d+)_p([01])_(ordinary|eager|fused)_sf(\d+)", folder.name
        )
        if not match:
            continue
        trial, pinned, mode, sf = match.groups()
        evidence = (folder / "operator_evidence.txt").read_text()
        peaks = re.findall(r"\[gpu_pool\].*?peak=(\d+) bytes", evidence)
        if not peaks:
            raise RuntimeError(f"Missing tracked pool statistics: {folder}")
        memory.append(
            dict(
                sf=sf,
                pinned=pinned,
                trial=trial,
                mode=mode,
                peak_bytes=max(map(int, peaks)),
            )
        )
    with (ROOT / "process_memory.csv").open("w") as output:
        writer = csv.DictWriter(
            output, fieldnames=["sf", "pinned", "trial", "mode", "peak_bytes"]
        )
        writer.writeheader()
        writer.writerows(memory)
    profiles = []
    for case in ["q12_sf10", "q14_sf1"]:
        for mode in ["ordinary", "eager", "fused"]:
            lines = (
                (ROOT / f"sirius_{case}_{mode}_pinned.kernels.txt")
                .read_text()
                .splitlines()
            )
            start = next(
                i for i, line in enumerate(lines) if line.startswith("Time (%),")
            )
            rows = list(csv.DictReader(lines[start:]))
            profiles.append(
                dict(
                    case=case,
                    mode=mode,
                    kernel_ms=sum(float(r["Total Time (ns)"]) for r in rows) / 1e6,
                    launches=sum(int(r["Instances"]) for r in rows),
                )
            )
    (ROOT / "profile_summary.json").write_text(json.dumps(profiles, indent=2) + "\n")
    library = (
        Path(__file__).resolve().parents[2]
        / "build/release/install/lib/libsirius.so.0.0.0"
    )
    resources = subprocess.check_output(
        ["cuobjdump", "--dump-resource-usage", str(library)], text=True
    ).splitlines()
    selected = []
    for index, line in enumerate(resources):
        if "Function" in line and "sirius2op" in line and "5probeE" in line:
            selected.extend(resources[index : index + 2])
    assert selected, "The fused probe's compiler resource declaration is missing"
    (ROOT / "lazy_kernel_resources.txt").write_text(
        "Compiler-declared resources; not measured occupancy or local-memory traffic.\n"
        + "\n".join(selected)
        + "\n"
    )
    print(f"Extracted {len(memory)} process peaks and {len(profiles)} captures")


if __name__ == "__main__":
    main()
