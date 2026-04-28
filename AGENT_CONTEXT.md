# Agent Context

## Project Background

- This repository is a downstream research-oriented fork of [siemens/powergym](https://github.com/siemens/powergym).
- The current focus is joint optimization of EV charging schedules and storage control under distribution-network power-flow constraints.
- Compared with upstream, the main changes are:
  - shifting control emphasis toward batteries and charging stations
  - introducing EV arrival and departure processes, waiting queues, and connection-point management
  - supporting both PPO-based and rule-based experiment workflows

## Environment Workflow

- Use the Conda environment declared in `environment.yml`: `dssgym-py312`.
- Never use Conda `base` or system-global Python for repository work.
- Preferred execution path: `conda run -n dssgym-py312 python ...` or an activated `dssgym-py312` shell.
- Do not execute `<env_path>\python.exe` directly for routine runs.

## Conda Stability Policy

- Do not treat `CONDA_NO_PLUGINS`, `CONDA_OVERRIDE_CUDA`, or `CONDA_REPORT_ERRORS` as required defaults.
- If `conda env list` or `conda info --envs` fails but `conda run -n dssgym-py312 ...` works, continue with `conda run`.
- Use the environment variables above only as a targeted workaround after a specific failure, and state that it is a workaround.

## Rebuild Workflow

- When the user requests reinstall or rebuild, use:
  1. `conda env remove -n dssgym-py312 -y`
  2. `conda env create -f environment.yml`
- Do not replace rebuild with `conda env update` unless the user explicitly requests an update workflow.

## Environment Guardrails

- Do not create Conda environments with `-p` or `--prefix` inside repository paths.
- Do not set `CONDA_PKGS_DIRS` or other package caches to repository paths.

## Codebase Landmarks

- Core source files live in `dssgym/*.py`.
- Root entry scripts are `ppo_agent.py`, `rules_agent.py`, and `test_results_analysis.py`.
- Treat `results/` and `本科论文/` as generated artifacts or experiment outputs unless the task is explicitly about analysis.

## Entry Points

- PPO train and test entrypoint: `ppo_agent.py`
- Rule-based evaluation entrypoint: `rules_agent.py`
- Result post-processing entrypoint: `test_results_analysis.py`

## Runtime Truths

- `dssgym/env_register.py` is the source of truth for environment wiring.
- Do not rely on the earlier per-environment `max_episode_steps=24` definitions in that file; the final applied value is `96` for every registered environment.
- The same override block also forces:
  - `reg_w=0.0`
  - `cap_w=0.0`
  - `soc_w=5.0`
  - `dis_w=0.0`
  - EV-station reward weights including `completion_w`, `connection_w`, `energy_w`, `voltage_w`, and `tf_capacity_w`
- `13Bus` is the safest default environment for EV charging work.
- Other systems still contain placeholder station metadata in `_STATION_INFO`.
- Environment names may use the suffix `_s<scale>`, such as `13Bus_cbat_s2.0`.
- `get_info_and_folder()` rescales `soc_w` by `scale ** 2`.

## EV Demand Resolution

EV demand CSV resolution order:

1. `--ev_demand_path` passed through runtime config
2. `DSSGYM_EV_DEMAND_<SYSTEM_NAME>`
3. `DSSGYM_EV_DEMAND_PATH`
4. `_EV_INFO` default in `dssgym/env_register.py`

- `get_info_and_folder(..., validate_ev_demand=True)` raises `FileNotFoundError` if no path resolves to a real file.

## CLI Quirks

- `ppo_agent.py` parses booleans as explicit string values such as `true` and `false`.
- `rules_agent.py` imports `ppo_agent.parse_arguments()` before adding its own parser, so inherited boolean flags also require explicit values.
- `python rules_agent.py --test_only` fails because the flag still expects a value.
- `--mode dss` is legacy and soft-deprecated.
- `ppo_agent.py` skips legacy DSS mode unless `--allow_legacy_dss true` is provided.

## Output Paths

- PPO training writes to `results/results_<timestamp>_<env_name>_<num_steps>/` under the current working directory.
- PPO evaluation writes `test_results_<timestamp>/`.
- If `model_path` is supplied for evaluation, the output directory is created under that model's parent result directory.
- Rule evaluation writes `test_results_rules_<timestamp>_<env_name>/` under the chosen output directory or the current working directory.
- `test_results_analysis.py` reads a results directory but saves generated PNG files to the current working directory instead of the results directory.

## Focused Verification

- There is no committed automated test suite or lint or typecheck command in this repository.
- Fast smoke checks that were already verified in this repo:
  - `conda run -n dssgym-py312 python -c "import dssgym; print('import ok')"`
  - `conda run -n dssgym-py312 python -c "from dssgym.env_register import get_info_and_folder; info,_=get_info_and_folder('13Bus_cbat'); print(info['ev_demand']); print(info['max_episode_steps']); print(info['bus_name'])"`
