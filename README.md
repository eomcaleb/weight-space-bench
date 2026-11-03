# weight-space-bench

**Can an LLM make a small language model better at math by editing its weights directly?**

Each model gets the same task as Experiment 8 of
[Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/). It reads the weights of
Qwen3.5-0.8B, finds out why the model fails at math, and edits the weights to fix it. It has no training and no
gradients. Afterwards the edited model is scored on MATH-500 and GSM8K. The model whose edit raises the score the most
wins.

## How a run works

1. **Edit (one model, through OpenRouter).** The model gets the paper's exact task message. It can act only by
   writing code blocks:
   - ` ```python `: runs in a sandboxed Python kernel over the weights (no network, no GPU, 900 s per run, at most 300 runs).
   - ` ```forward `: runs Qwen3.5-0.8B on practice problems on the GPU. There is a fixed budget of forward passes
     (10 by default). A request that fails does not count.

   When the passes are used up, the tensors in `/work/edited` are submitted.
2. **Evaluate (local GPU).** The submitted tensors replace the originals as bfloat16. The model then answers MATH-500
   (4 samples each) and GSM8K (2 samples each) with the paper's sampling settings and a 32,768-token limit.
3. **Leaderboard.** Score = mean of the MATH-500 and GSM8K accuracies. The gain is measured against the unedited
   model, which you evaluate on the same machine.

## Isolated and standardized

- **No tools for the editing model.** The API request has no `tools` field, no plugins and no web search (`:online`
  slugs are refused). Every reply is audited for tool calls and web citations. A reply that has them is discarded.
  If that happens 3 times in a row, the run is marked invalid and is not ranked.
- **Sandboxed code.** The kernel runs in a Docker container with `--network none`. It sees only the model
  (read-only), the practice problems (read-only) and its work folder. The test sets are never mounted.
- **Same conditions for every model.** Every model gets the same system line and task text, the same budgets and
  limits, the same reasoning effort and the same evaluation protocol. A hash of these settings is saved with each run.
  The leaderboard ranks only runs that match the current settings.
- **Everything is logged** in `results/<model>/`: every message, reply, code run and forward pass, plus the
  submission and the final answer.

## Quick start

You need Linux (or WSL2 on Windows), an NVIDIA GPU, Docker, and an [OpenRouter](https://openrouter.ai) API key.

```bash
git clone https://github.com/eomcaleb/weight-space-bench && cd weight-space-bench
pip install -r requirements.txt
docker build -f docker/Dockerfile.kernel -t weight-space-bench-kernel .
export OPENROUTER_API_KEY=sk-or-...

python -m weight_space_bench setup        # download Qwen3.5-0.8B (pinned revision), practice set, MATH-500, GSM8K
python -m weight_space_bench check        # smoke test: kernel, GPU passes, grading (no API calls)
python -m weight_space_bench baseline     # score the unedited model
python -m weight_space_bench reference    # optional: score the paper's own edit (should be close to 48.1 / 67.1)
python -m weight_space_bench run all      # editing sessions for every model in config/models.toml
python -m weight_space_bench eval all     # score each submission
python -m weight_space_bench leaderboard  # -> results/leaderboard_full.md
```

Use `python -m weight_space_bench all` to do baseline, runs, evals and leaderboard in one go. If a run or an
evaluation is interrupted, start the same command again and it resumes.

## Models

The models are set in `config/models.toml`. The default lineup is 6 models:

| Model | OpenRouter slug | Reasoning effort |
|---|---|---|
| GPT-6.1 Sol | `openai/gpt-6.1-sol` | max |
| DeepSeek V4.1 Flash 1017 | `deepseek/deepseek-v4.1-flash-1017` | max |
| Kimi K3.1 Pro | `moonshotai/kimi-k3.1-pro` | high |
| Muse Wonder | `meta/muse-wonder` | max |
| Claude Opus 5.5 | `anthropic/claude-opus-5.5` | max |
| Claude Sonnet 5.5 | `anthropic/claude-sonnet-5.5` | max |

Some providers may reject "max". In that case the run falls back to "high", and the leaderboard shows the effort that
was actually used. Run one model with `python -m weight_space_bench run claude-opus-5.5`.

To add a model, add another `[[model]]` block.

## Settings

The settings are in `config/benchmark.toml`:

- Budget: `forward_passes` (the paper ran 10, 25 and 50).
- Cost cap per run: `max_cost_usd`.
- Kernel memory.
- `--profile quick`: a cheaper evaluation that uses about a quarter of the tokens. Compare quick scores only with
  other quick scores.

## Cost and time

- **API:** a run is about 75 to 160 replies with long contexts. Expect tens to a few hundred dollars per model,
  depending on its price. The cap is set by `max_cost_usd`.
- **GPU:** the unedited model often thinks until it hits the 32,768-token limit, so generation is slow. On a 12 GB card
  (RTX 3060), expect roughly an hour per generation pass. A full evaluation takes most of a day. The paper used
  data-centre GPUs.

## Future benchmarks

For now there is one test, the math weight edit. Other experiments from the paper could be added later as separate
tests in the same style:

- **Which game?** Name the game a small PPO policy was trained on, from its weights (exp 1). Scored by accuracy.
- **Which shape?** Name the shape a small diffusion model draws (exp 2). Scored by accuracy.
- **Write a policy from scratch.** Write MinAtar policy weights from zeros (exp 5). Scored by mean game score
  against PPO.
- **Which language?** Name the training text and find the copying heads of a char transformer (exp 3). Scored by
  accuracy and head F1.

The other experiments are left out. Exp 4 needs a physics simulator the paper did not release. Exp 6 needs Go
opponents and an evaluation harness that were not released. Exp 7's weights were not released.

## Credits

The task, prompts, practice set, grading code and reference edit come from the authors'
[released logs](https://github.com/RikiyaT/lm_understand-logs) (logs and data CC BY 4.0, scripts MIT). The editing
harness itself was not released, so this repository rebuilds it from the prompts.

## Contributing and license

Contributions are welcome: new models, results, bug fixes and new tests. See [CONTRIBUTING.md](CONTRIBUTING.md).

The code is under the [MIT License](LICENSE). Files taken from other projects keep their own licenses; see
[THIRD_PARTY.md](THIRD_PARTY.md).
