# External baselines and corruption tools

## LGA (not bundled)

Official source: https://github.com/dongbeank/LGA
Follow the upstream README and applicable permissions before obtaining or using
the code. This repository does not grant permission to redistribute LGA.

If permitted, obtain it outside this repository, follow its dependency/data
instructions, and set the absolute path before using the optional queue:

```bash
export FCD_LGA_ROOT=/absolute/path/to/your/LGA
python scripts/experiments/gpu_queue_traffic_lga.py --help
```

The queue explicitly checks this path when LGA jobs are requested. Its --no-lga
option does not require LGA. --help is read-only; running the queue launches jobs.
Record the upstream commit and local configuration for a new experiment. The
current upstream version is not asserted to reproduce a historical result.

## Legacy LGA-bundled TSRBench (not bundled)

The former top-level TSRBench/ and LGA_official/TSRBench/ code are omitted because
their specific redistribution terms were not confirmed. Consult the LGA upstream
README about that historical tool. Do not substitute a newer noise generator
while claiming identical historical corrupted data or scores.

## Standalone TSRBench (MIT)

Official source: https://github.com/dongbeank/TSRBench
The MIT-licensed source snapshot in TSRBench_official/ retains its LICENSE.
For a new experiment, use the included snapshot in a separate environment:

```bash
python -m pip install -e ./TSRBench_official
python -m tsrbench --help
```

Upstream also documents installing `tsrbench` from PyPI. Record the exact version,
seed, data preprocessing and generator parameters; installation of the latest
package alone does not identify the historical protocol.
