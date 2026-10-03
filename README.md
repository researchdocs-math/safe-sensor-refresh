# Safe Sensor Refresh for Edge Agents

A reproducible implementation of cost-aware sensor refreshing before physical action authorization. The package includes positive linear process simulators, robust reachable-tube certificates, exact and greedy sensor selectors, a local language-model interface, recorded proposal traces, and reference results.

## Quick start

Use Python 3.12. The deterministic simulations and recorded-proposal replay run on a CPU.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_all.py --workers 10
```

`run_all.py` executes the mathematical checks, reproduces all five simulation suites, replays the local-agent episodes, computes episode-level summaries and paired bootstrap intervals, and compares the numeric output with the reference results. The final `results/reproduced/verification.json` records each comparison. Floating-point values are checked with absolute and relative tolerances of 1e-12.

The reference environment is an Apple M3 Pro with 12 CPU cores, 18 GB memory, Python 3.12.1, NumPy 1.26.3, SciPy 1.11.4, and pandas 2.2.1. Ten worker processes use one numerical-library thread each. Choose fewer workers on smaller machines. Runtime depends on hardware and system load.

## Experiment suites

| Suite | Design | Episodes |
|---|---|---:|
| Main | Two regimes, nine policies, 512 seeds, 72 proposals | 9,216 |
| Deadlines | Two regimes, six policies, five deadlines, 128 seeds | 7,680 |
| Persistent disturbance | Two regimes, nine policies, 256 seeds | 4,608 |
| Loss and misspecification | Two regimes, three policies, five settings, 256 seeds | 7,680 |
| Post-query recheck ablation | Two regimes, three policies, three settings, 128 seeds | 2,304 |
| Local-agent integration | Two regimes, three guards, four seeds, 36 proposals | 24 |

The regimes are synthetic normalized storage-like and thermal-like systems. Every plant parameter is defined in `src/saferefresh.py`. Protocol dimensions and seed offsets are in `config/study.json`. Safety is conditional on the modeled dynamics, measurement and disturbance bounds, timestamps, and invariant backup. The misspecification suite deliberately violates the disturbance assumption.

The main selector enumerates nonempty subsets at eight sensors. `choose_milp` uses one exact binary covering program per integer communication-delay class. `tests/validate.py` compares robust support values with direct worst-case rollouts and checks selection against independently enumerated subsets.

## Results and measurements

`results/reference/` contains episode-level CSV files, aggregate statistics, environment information, and original run records. Each CSV row identifies the regime, policy, seed, deadline, disturbance setting, communication cost, authorization count, target error, and safety/containment diagnostics. The `water` and `thermal` domain identifiers denote the storage-like and thermal-like synthetic regimes, respectively.

- `cost`: total integer communication units, including lost queries.
- `accepted`: number of proposed held inputs authorized by the policy.
- `violation_episode`: one if any individual or aggregate state constraint is violated.
- `tracking_mae`: mean absolute target error during execution ticks.
- `wait_ticks`: process ticks spent acquiring measurements under backup.
- `forecast_failures`: selected query batches after which the proposal is still uncertified.
- `containment_failures`: ticks at which the true simulated state leaves the maintained interval.

The bootstrap uses paired episode seeds, 10,000 resamples, and a fixed analysis seed. Entire episodes are resampled to retain temporal dependence. Regenerate timing distributions with:

```bash
python run_all.py --workers 10 --benchmark
```

`scaling.csv` records raw solver durations, feasibility, and query costs. Runtime values vary by machine; feasibility and selected forecast costs are deterministic. The first mixed-integer call includes solver initialization.

## Local GPU inference

The 864 recorded proposals, prompt inputs, model outputs, and inference measurements are in `data/local_agent_inference.json`. `data/local_agent_tapes.json` stores policy-specific action sequences and trajectories. The standard replay reproduces the measured physical outcomes exactly from these proposals.

To regenerate proposals on Apple silicon, install the optional MLX environment and download the pinned model revision:

```bash
python -m pip install -r requirements-mlx.txt
python scripts/download_model.py
python scripts/run_local_agent.py --model models/Qwen3-1.7B-4bit --out results/local_new
```

The model is `mlx-community/Qwen3-1.7B-4bit`, revision `3b1b1768f8f8cf8351c712464f906e86c2b8269e`. Model-file hashes are recorded in `config/model.json`. Its weights are distributed under Apache-2.0 by the model publisher. Inference uses non-thinking mode, greedy decoding, a 96-token limit, and a schema-checked actuator interface. `results/reference/local_agent_run.json` contains the system prompt and inference settings. The simulator advances in lock-step; reported inference latency is a separate wall-clock measurement. Regenerated model responses can vary with runtime/library implementations, while the recorded-proposal replay fixes the experiment's action sequence.

## Layout

```text
safe-sensor-refresh/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── requirements-mlx.txt
├── run_all.py
├── config/
├── src/
├── scripts/
├── tests/
├── data/
└── results/reference/
```

The original implementation is available under the MIT license. Citation metadata is provided in `CITATION.cff`.
