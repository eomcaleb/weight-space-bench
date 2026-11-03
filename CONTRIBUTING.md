# Contributing

Contributions are welcome: bug fixes, new models, results, and new tests.

## Ground rules

- **Keep runs comparable.** Anything that changes what a model sees or how it is scored changes the benchmark. That
  includes the prompts, the system message, the number of attempts, the episodes, the seeds and the edits reader.
  Such changes need a new benchmark version and a note in the README. The settings hash in each run's `run.json`
  makes the change visible, and the leaderboard ranks only runs with the current hash.
- **Keep models isolated.** A model gets no tools: no web search, no plugins, no code execution. It answers in one
  reply.
- **Keep it simple.** Small, readable changes are best. Explain the reason in the pull request.

## Ways to help

- **Add a model.** Add a `[[model]]` block to `config/models.toml` with an OpenRouter slug you have checked.
- **Submit results.** Open a pull request with your `results/minatar/<model>/` folder (prompts, replies, networks,
  scores). Include the output of `python -m weight_space_bench reference` from your machine, which shows that your
  scorer reproduces the paper.
- **Add a test.** See "Future tests" in the README. Put each new test in its own subpackage (like `minatar/` and
  `math_edit/`) with its own config file.
- **Fix bugs.** Open an issue with the command, the error and your environment.

## Development

```bash
pip install -r requirements.txt
python -m weight_space_bench reference     # scores the paper's networks: a quick end-to-end check of the scorer
```

The run and leaderboard code can be tested without API calls by replacing `task.OpenRouter` with a fake client that
returns saved replies (for example the paper's replies in `weight_space_bench/minatar/assets/reference/`).

By contributing, you agree that your contributions are licensed under the MIT License of this repository.
