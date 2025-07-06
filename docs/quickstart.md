# Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Run a single-task benchmark
python scripts/run_benchmark.py --modes heuristic,groundtruth \
    --max_tasks 5 --asm_alpha 0.1 --asm_decay 0.999

# Train controller on 400 tasks saving assemblies
python scripts/train_controller.py --max_tasks 400 \
    --save_assemblies weights/assemblies.json

# Hyper-parameter grid-search
python scripts/grid_search.py --alphas 0.05 0.1 0.2 --decays 0.99 0.999
```
