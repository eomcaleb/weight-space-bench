# Third-party material

Some files in this repository come from other projects and keep their own licenses.

| Path | Source | License |
|---|---|---|
| `weight_space_bench/assets/task_template.txt`, `task_notes.txt`, `notices/` | Task message and experimenters' notes of Experiment 8, from the [authors' released logs](https://github.com/RikiyaT/lm_understand-logs) of *Sparks of Weight-Space Programming* (notes lightly merged) | CC BY 4.0 |
| `weight_space_bench/assets/reference/` | Astra's 50-pass weight edit (exact diff) and the published scores, from the same release | CC BY 4.0 |
| `weight_space_bench/assets/qwen35_text/` | Config, generation config, tokenizer config and chat template of the text-only Qwen3.5-0.8B, from the same release (derived from [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B)) | Apache-2.0 (Qwen) / CC BY 4.0 |
| `weight_space_bench/assets/practice/` | Practice problems as packaged in the same release: GSM8K train ([openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k), MIT), MATH train ([hendrycks/math](https://github.com/hendrycks/math), MIT) and AIME 1983-2023 problems | as stated by each source |
| `weight_space_bench/mathqa.py`, `weight_space_bench/st_numpy.py` | Grading and safetensors helpers from the same release (the answer extraction follows hendrycks/math) | MIT |

The model weights (Qwen3.5-0.8B, Apache-2.0) and the evaluation sets (MATH-500, GSM8K test) are not stored here;
`python -m weight_space_bench setup` downloads them from the Hugging Face Hub.
