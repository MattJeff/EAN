# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)

## [1.0.0] - 2025-07-06
### Added
- Assembly persistence with EMA update & exponential forgetting.
- CLI options `--load_assemblies`, `--save_assemblies`, `--asm-alpha`, `--asm-decay`.
- Analysis scripts: `analysis/dump_assemblies.py`, interactive notebook, mini `grid_search.py`.
- `clean_outputs.py` helper.
- Extensive unit tests (>87% coverage) and GitHub Actions pipeline with Codecov upload.
- Documentation updates, architecture diagram, Quickstart section.

### Changed
- `PolicyScheduler` parameterised for assembly hyper-parameters.
- Sorted `AssemblyStore` by strength.

### Removed
- Legacy artefacts in `weights/` & `bench_results/` by clean script.

### Fixed
- Multiple minor lint & docstring issues; CI now enforces *pydocstyle*.
