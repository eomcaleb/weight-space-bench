# Contributing

Contributions are welcome: bug fixes, new models, results, and new tests.

## Ground rules

- **Keep runs comparable.** Anything that changes what the editing model sees or can do (the task text, budgets,
  sandbox limits, the forward-pass tool, sampling, evaluation) changes the benchmark. Such changes need a new
  benchmark version and a note in the README. The settings hash in each run's `state.json` makes the change visible.
- **Keep the editing model isolated.** It must have no tools other than the ```python and ```forward blocks the
  harness runs. That means no web search, no plugins and no network in the kernel.
- **Keep it simple.** Small, readable changes are best. Explain the reason in the pull request.

## Ways to help

- **Add a model.** Add a `[[model]]` block to `config/models.toml` with an OpenRouter slug that you have checked.
- **Submit results.** Open a pull request with your run's `results/<model>/state.json`, `final_answer.md`,
  `submission/` manifest and `eval/<profile>/scores.json`. Include your GPU, vLLM version and the settings hash.
  Please also include the `reference` score for your machine, which shows that your evaluation reproduces the paper.
- **Add a test.** See "Future benchmarks" in the README. Put each new test in its own module with its own README.
- **Fix bugs.** Open an issue with the command, the error and your environment.

## Development

```bash
pip install -r requirements.txt
python -m weight_space_bench check      # needs Linux + an NVIDIA GPU
```

The kernel, agent loop, forward-tool validation and leaderboard can be tested without a GPU by faking the API and the
GPU worker. The tests made during development did this.

By contributing, you agree that your contributions are licensed under the MIT License of this repository.
