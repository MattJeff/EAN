# HyperCubeX EAN

[![CI](https://github.com/MattJeff/EAN/actions/workflows/python-ci.yml/badge.svg)](https://github.com/MattJeff/EAN/actions) [![codecov](https://codecov.io/gh/MattJeff/EAN/branch/main/graph/badge.svg)](https://codecov.io/gh/MattJeff/EAN)

> Emergent Assembly Networks – rotation & mirror reasoning with GPU acceleration

![architecture](docs/diagram_arch.svg)

HyperCubeX EAN (HCX-EAN) is a research framework that explores **Emergent Assembly Networks** – minimal neuron-like agents able to solve Abstract Reasoning Corpus (ARC) tasks via self-organisation.

## Features

| Module | Status |
|--------|--------|
| Rotation & Mirror connectors | ✅ |
| BatchPolicyScheduler (multi-task) | ✅ |
| Shared REINFORCE baseline | ✅ |
| GPU backend (PyTorch – CUDA/MPS) | ✅ prototype |
| Monitoring (CSV/JSON + GIF/MP4) | ✅ |
| Robustness (synaptic noise + refractory) | ✅ |
| CI/CD (lint, tests, coverage) | ✅ |
| Docs & Diagrams | ⏳ |

## Quickstart

```bash
# clone & install
pip install -r requirements.txt  # core deps
pip install -r requirements-dev.txt  # dev / CI deps

# run tests
pytest -q

# benchmark CPU vs GPU (CUDA/MPS)
export EAN_BACKEND=torch
python benchmarks/bench_ean.py --grid 100 --ticks 5000 --backend torch
python benchmarks/bench_ean.py --grid 100 --ticks 5000 --backend cpu
```

## Memory Quickstart

```bash
# run controller benchmark with memory persistence
python scripts/run_benchmark.py \
  --data_dir data/training \
  --modes heuristic,controller \
  --load_assemblies weights/assemblies.json \
  --save_assemblies weights/assemblies.json

# advanced: GPU benchmark
export EAN_BACKEND=torch
python benchmarks/bench_ean.py --grid 100 --ticks 5000 --backend torch
python benchmarks/bench_ean.py --grid 100 --ticks 5000 --backend cpu
```

## Running an ARC curriculum

```bash
python experiments/curriculum_rotation3x3.py \
  --ticks 3000 \
  --batch 8 \
  --backend torch \
  --noise-std 0.05 \
  --logger run.csv --visualiser spikes.gif
```

CSV & JSON-Lines logs will be generated; an animated GIF of energy & spikes is exported.

## Architecture Overview

```text
┌────────────┐      ┌────────────┐      ┌────────────┐
│  Teachers  │ ◀──▶ │  Adapters  │ ◀──▶ │ Assemblies │
└────────────┘      └────────────┘      └────────────┘
       ▲                                   ▲
       │                                   │
       │          ┌────────────┐          │
       └─────────▶│  Network   │◀─────────┘
                  └────────────┘
                       ▲
                       │  backend = CPU | Torch
                       ▼
                  ┌────────────┐
                  │  Backend   │
                  └────────────┘
```

## Contributing

1. Create branch `feature/<name>`.
2. Ensure `pytest -q` & `flake8` pass.
3. Push & open PR – GitHub Actions will run lint, tests & coverage.

## License

MIT © 2025 HyperCubeX Contributors
