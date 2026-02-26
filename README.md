
# Breaking-MT

This project explores adversarial rewriting for machine translation (MT) using reinforcement learning. Given an English sentence, we fine-tune a LLM to generate a rewrite that is intentionally harder for MT systems to translate.

We fine-tune Llama-3.1-8B-Instruct with Group Relative Policy Optimization (GRPO) and LoRA, optimizing a translation-difficulty estimator ([Sentinel](https://huggingface.co/Prosho/sentinel-src-25)) while enforcing constraints on grammaticality, semantic similarity and length. The model learns to increase translation difficulty without resorting to degenerate strategies such as gibberish or excessive length.

**Report:** Further details about the methodology, experiments, reward design and analysis can be found in our full report: [Breaking-MT (PDF)](report_breaking-MT.pdf)

### Setup

This project uses **Conda** to manage dependencies and **Unsloth** for efficient fine-tuning.



To create the environment, we provide `environment.yml` that handles both Conda and Pip dependencies

```bash
conda env create -f environment.yml
conda activate mtbreaker
```
---


### 1. Training (GRPO Fine-Tuning)

To start the adversarial fine-tuning process using Group Relative Policy Optimization (GRPO) on Llama-3.1-8B:

```bash
python src/model_training.py
```

Run the help command to check all available parameters. The current default is our typical run configuration. For W&B logging, add your API key to the repo or login manually.

### 2. Evaluation

To evaluate the fine-tuned model against the base model using the Sentinel metric and validity checks:

```bash
python src/eval_grpo_harder_translate.py
```

You can either train a model and then ensure to point the evaluation script to the correct checkpoint path, or you can use our checkpoint from HuggingFace ``matafix/mtbreaker-5000`` (current default). 
