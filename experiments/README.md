# Join–aggregation research artifacts

The [shared cuDF evaluator study](join_map_aggregate/REPORT.md) follows the
[custom lazy-fusion study](fused_join_aggregate/REPORT.md). Both retain negative
results, Python plotting scripts, figures, raw timing tables, build metadata,
semantic checks and compact execution evidence. Experimental engine paths are
disabled by default.

## Retained evidence and cleanup

Repeated SQL and CPU reference files are stored once by content in each study's
`results/reference_fixtures.json.gz`. Its `files` index preserves the original
relative paths, content hashes and compression flags. The `contents` dictionary
holds each unique text. The archive was read back and checked before removing
the repeated copies. Per-run timing/results JSON, environment metadata, operator
counts and extracted execution/plan evidence remain at their original paths.

Validate or restore the reference fixtures from the repository root:

```bash
python3 experiments/reference_fixtures.py experiments/fused_join_aggregate/results check
python3 experiments/reference_fixtures.py experiments/join_map_aggregate/results check
# Restore the original per-run fixture paths when needed:
python3 experiments/reference_fixtures.py experiments/fused_join_aggregate/results restore
python3 experiments/reference_fixtures.py experiments/join_map_aggregate/results restore
```

Restoring a compressed JSON reference preserves its contents; the gzip header
can differ. Fresh measurement runs create their own fixtures normally. Plotting
uses retained measurement/evidence files and does not require fixture restoration.

Superseded development pilots, duplicate console timing summaries, empty stderr
files, raw logs, generated TPC-H databases/Parquet, experiment executables, Python
caches and raw profiler captures were deleted. Dataset manifests, generators,
profiler summaries and sanitizer diagnostics remain. Generate datasets again
using the commands in the reports before rerunning SQL measurements.

## cuCascade dependency

The submodule gitlink remains at the original committed revision. Its KvikIO
changes are retained in
[cucascade-kvikio.patch](fused_join_aggregate/cucascade-kvikio.patch), rather than
an unpublished submodule commit. The
[research build script](fused_join_aggregate/build_sirius_kvikio.sh) applies that
patch before building. A clean checkout therefore needs this script to reproduce
the KvikIO-enabled research build; applying it modifies the submodule worktree.
