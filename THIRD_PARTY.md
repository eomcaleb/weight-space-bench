# Third-party material

Some files in this repository come from other projects and keep their own licenses.

## Test 1: write MinAtar networks from scratch

| Path | Source | License |
|---|---|---|
| `weight_space_bench/minatar/assets/prompts/*.txt` | The paper's prompts for Experiment 5, from the [authors' released logs](https://github.com/RikiyaT/lm_understand-logs). Each prompt contains one game's source file from [MinAtar](https://github.com/kenjyoung/MinAtar) 1.0.15 (Kenny Young and Tian Tian), unchanged, with its author header | Prompt text: CC BY 4.0. MinAtar source: GPL-3.0 (full text in `weight_space_bench/minatar/assets/LICENSE-MinAtar`) |
| `weight_space_bench/minatar/assets/reference/` | GPT-6 Astra's replies and networks, and the paper's evaluation (PPO baselines), from the same release | CC BY 4.0 |

MinAtar itself is installed from PyPI (`MinAtar==1.0.15`, GPL-3.0); this repository does not include its package.

## Optional test: edit a language model for math

| Path | Source | License |
|---|---|---|
| `weight_space_bench/math_edit/assets/task_template.txt`, `task_notes.txt`, `notices/` | Experiment 8's task message and the experimenters' notes, from the same release (notes lightly merged) | CC BY 4.0 |
| `weight_space_bench/math_edit/assets/reference/` | Astra's 50-pass weight edit (exact diff) and the published scores, from the same release | CC BY 4.0 |
| `weight_space_bench/math_edit/assets/qwen35_text/` | Config, generation config, tokenizer config and chat template of the text-only Qwen3.5-0.8B, from the same release (derived from [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B)) | Apache-2.0 (Qwen) / CC BY 4.0 |
| `weight_space_bench/math_edit/assets/practice/` | Practice problems as packaged in the same release: GSM8K train ([openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k), MIT), MATH train ([hendrycks/math](https://github.com/hendrycks/math), MIT) and AIME 1983-2023 problems | as stated by each source |
| `weight_space_bench/math_edit/mathqa.py`, `st_numpy.py` | Grading and safetensors helpers from the same release (answer extraction follows hendrycks/math) | MIT |

The model weights (Qwen3.5-0.8B, Apache-2.0) and the evaluation sets (MATH-500, GSM8K test) are not stored here;
`python -m weight_space_bench math-edit setup` downloads them from the Hugging Face Hub.
