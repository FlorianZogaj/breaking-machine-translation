
# Breaking-MT
Reinforcement Learning Framework for Adversarial Machine Translation Generation

---
<img src="imgs/Architecture.png" alt="Architecture" width="80%">

## Overview

Breaking-MT uses **Proximal Policy Optimization (PPO)** to train language models to generate English sentences that are difficult for machine translation (MT) systems to translate, while ensuring the sentences remain grammatical, natural, and concise.

The framework directly optimizes a language model policy using reinforcement learning with a composite reward function that balances:
- **Translation Difficulty**: How hard the generated sentence is to translate (measured via COMET QE)
- **Linguistic Quality**: Grammaticality and naturalness (measured via LM perplexity)
- **Constraints**: Length bounds and lexical diversity

Unlike traditional supervised fine-tuning approaches, this PPO-based method can directly optimize non-differentiable objectives like translation quality estimation and MT system outputs.

---

## Key Features

- **End-to-End RL Training**: Direct policy optimization using PPO without requiring supervised data
- **Multi-Modal Reward Function**: Combines translation difficulty (COMET QE), linguistic naturalness (LM perplexity), and constraint satisfaction
- **Reference-Free Quality Estimation**: Uses COMET-KIWI for translation quality without requiring reference translations
- **Advanced PPO Implementation**:
  - Mixed precision training (FP16/FP32) for memory efficiency
  - Gradient checkpointing to reduce memory footprint
  - Adaptive KL penalty to prevent policy collapse
  - Configurable device allocation for multi-model inference
  - Mini-batch processing for stable training
- **Weights & Biases Integration**: Comprehensive experiment tracking and model versioning
- **SLURM Support**: Production-ready scripts for HPC cluster deployment
- **Flexible Model Support**: Works with GPT-2, Qwen3-0.6B, and other causal language models

---

## Environment Setup
```bash
conda env create -f env.yaml
conda activate mtbreaker
```

Note: First run will download pretrained models:
- Qwen/Qwen3-0.6B-Base or GPT-2 (policy model)
- Helsinki-NLP/opus-mt-en-de (EN→DE translator)
- Unbabel/wmt22-cometkiwi-da (COMET QE scorer)
- distilgpt2 (linguistic verifier)


---

## Project Structure

```
breaking-MT/
├─ env.yaml                        # Conda environment specification
├─ requirements.txt                # Pip dependencies with exact versions
├─ data/
│  ├─ seeds.txt                    # English seed sentences for training
│  └─ method2_results.jsonl        # (Optional) Analysis output from method2.py
├─ src/
│  ├─ RL_ppo_training.py           # Main PPO training script ⭐
│  ├─ translate.py                 # English→German translation (Helsinki-NLP/opus-mt-en-de)
│  ├─ scorers.py                   # COMET QE (wmt22-cometkiwi-da) + LM verifier (distilgpt2)
│  ├─ constraints.py               # Length & lexical diversity scoring
│  ├─ losses.py                    # Reward function implementation
│  ├─ method2.py                   # (Optional) Analysis pipeline for reward computation
│  ├─ gen_model.py                 # (Optional) Standalone LLM editor
│  ├─ sft_lora.py                  # (Optional) Supervised fine-tuning with LoRA
│  └─ __init__.py
├─ scripts/
│  ├─ run.sh                       # SLURM job script for PPO training
│  ├─ verify.sh                    # SLURM script for model verification
│  └─ test_model.sh                # SLURM script for model testing
├─ test_scripts/
│  ├─ test_model_inference.py      # Test script for model inference
│  ├─ check_tokenizer.py           # Tokenizer verification utility
│  └─ verify_gpt2_generation.py    # Generation quality verification
├─ checkpoints/
│  └─ gpt2-ppo-method2/            # Saved PPO policy checkpoints
└─ README.md
```

---

## Quick Start

### 1. Prepare Training Data

Create `data/seeds.txt` with one English sentence per line. These should be simple, easily-translatable sentences that the model will learn to make more difficult.

Example seeds:
```
A beautiful bird sings happily.
He saw her duck.
The bank is by the river.
The weather is nice today.
```

### 2. Run PPO Training

**Basic training (recommended for first run):**
```bash
python -m src.RL_ppo_training \
  --seeds data/seeds.txt \
  --k 200 \
  --steps 300 \
  --batch_size 2 \
  --temperature 0.7
```

**SLURM cluster training:**
```bash
sbatch scripts/run.sh
```

The model will:
1. Load seed sentences from `data/seeds.txt`
2. Initialize the policy (Qwen3-0.6B or GPT-2)
3. For each PPO step:
   - Generate edited sentences using the current policy
   - Translate originals and edits to German
   - Compute rewards (difficulty, naturalness, constraints)
   - Update the policy to maximize rewards
4. Save checkpoints every 20 steps to `checkpoints/`

---

## PPO Training Details

### Core Training Script: `RL_ppo_training.py`

This is the main entry point for training. It implements a complete PPO pipeline with:

#### Training Loop Architecture

```
For each PPO step:
├─ Sample batch of seed sentences
├─ Build instruction prompts
├─ Generate edits with current policy (autoregressive LM)
├─ Compute rewards via external scorers:
│  ├─ Translate original & edit to German (MT model)
│  ├─ Score translation difficulty (COMET QE)
│  ├─ Score linguistic quality (LM verifier)
│  └─ Score constraints (length, diversity)
├─ Calculate composite reward: reward = -L
├─ PPO update step:
│  ├─ Compute advantages
│  ├─ Update policy with clipped objective
│  ├─ Apply KL penalty
│  └─ Update value function
└─ Log metrics to WandB and save checkpoints
```

### Command-Line Arguments

#### Essential Parameters

- `--seeds` (str): Path to seed sentences file (default: `data/seeds.txt`)
- `--k` (int): Number of seeds to use for training (default: 500)
- `--steps` (int): Number of PPO training steps (default: 200)
- `--batch_size` (int): Number of prompts per PPO step (default: 2)

#### Model Configuration

- `--base_model` (str): Base LM for policy (default: `Qwen/Qwen3-0.6B-Base`)
  - Alternatives: `gpt2`, `gpt2-medium`, etc.
- `--save_dir` (str): Checkpoint output directory (default: `checkpoints/gpt2-ppo-method2`)

#### Generation Parameters

- `--gen_max_new_tokens` (int): Max tokens to generate per sample (default: 64)
- `--temperature` (float): Sampling temperature (default: 0.7)
  - Qwen3 recommended: 0.7 for standard mode, 0.6 for thinking mode
  - Higher = more diverse outputs
- `--top_p` (float): Nucleus sampling threshold (default: 0.8)
  - Qwen3 recommended: 0.8 for standard mode, 0.95 for thinking mode

#### Reward Function Weights

- `--x` (float): Weight for difficulty delta (default: 1.0)
- `--y` (float): Weight for constraint score (default: 0.3)
- `--z` (float): Weight for verifier score (default: 0.3)
- `--f` (str): Transformation for difficulty delta (default: `none`)
  - `relu`: Only reward positive improvements
  - `sigmoid`: Smooth transformation
  - `none`: Use raw delta

#### Device Allocation

- `--mt_device` (str): Device for MT translator (choices: `cpu`, `cuda`, default: `cuda`)
- `--lm_verifier_device` (str): Device for LM verifier (choices: `cpu`, `cuda`, default: `cuda`)
- `--comet_accelerator` (str): Accelerator for COMET QE (choices: `cpu`, `gpu`, default: `gpu`)

#### Experiment Tracking

- `--wandb_project` (str): WandB project name (default: `mt-breaker-ppo`)
- `--wandb_run_name` (str): WandB run name (default: auto-generated)
- `--no_wandb`: Disable WandB logging
- `--seed` (int): Random seed for reproducibility (default: 42)

### Example Commands

**Full GPU training (11GB VRAM):**
```bash
python -m src.RL_ppo_training \
  --seeds data/seeds.txt \
  --k 500 \
  --base_model "Qwen/Qwen3-0.6B-Base" \
  --steps 300 \
  --batch_size 2 \
  --temperature 0.7 \
  --mt_device cuda \
  --lm_verifier_device cuda \
  --comet_accelerator gpu \
  --wandb_project my-mt-breaker \
  --wandb_run_name qwen3-experiment-1
```

**Memory-efficient training (8GB VRAM):**
```bash
python -m src.RL_ppo_training \
  --seeds data/seeds.txt \
  --k 200 \
  --base_model "gpt2" \
  --steps 200 \
  --batch_size 1 \
  --gen_max_new_tokens 48 \
  --mt_device cpu \
  --lm_verifier_device cpu \
  --comet_accelerator cpu
```

**Quick debug run:**
```bash
python -m src.RL_ppo_training \
  --seeds data/seeds.txt \
  --k 50 \
  --steps 10 \
  --batch_size 2 \
  --no_wandb
```

---

## Reward Function

The reward function is implemented in [src/losses.py](src/losses.py) and computed during training:

### Mathematical Formulation

```
L = 1 - [ x * f(Δde) + y * constraint(s) + z * verify(s) ] / (x + y + z)

Reward = -L  (higher is better for PPO)
```

Where:
- **Δde = de(edit, t_edit) - de(seed, t_seed)**: Change in translation difficulty
  - **de(s,t) = 1 - QE(s,t)**: Difficulty score from COMET QE
  - Higher Δde = edit is harder to translate than seed
- **constraint(s)**: Length and lexical diversity score
  - Rewards sentences within length bounds (4-1024 tokens)
  - Penalizes low lexical diversity (repeated words)
  - Returns value in [0, 1]
- **verify(s)**: Linguistic naturalness score
  - Based on LM perplexity using distilgpt2
  - Higher score = more grammatical/natural
- **f()**: Optional transformation function
  - `relu`: max(0, Δde) - only reward improvements
  - `sigmoid`: 1/(1 + exp(-Δde)) - smooth squashing
  - `none`: raw Δde - allow negative rewards

### Component Models

| Component | Model | Role | Device |
|-----------|-------|------|--------|
| Policy (trainable) | Qwen3-0.6B / GPT-2 | Generates difficult sentences | GPU |
| Reference (frozen) | Same as policy | KL divergence baseline | GPU |
| MT Translator | Helsinki-NLP/opus-mt-en-de | EN→DE translation | GPU/CPU |
| Quality Estimator | Unbabel/wmt22-cometkiwi-da | Translation difficulty | GPU/CPU |
| Verifier | distilgpt2 | Linguistic quality | GPU/CPU |

### Reward Components Breakdown

**Example reward computation:**
```
Seed: "The cat sat on the mat."
Edit: "The feline perched atop the woven floor covering, observing."

1. Translation Difficulty:
   - de(seed, t_seed) = 0.15  (easy to translate)
   - de(edit, t_edit) = 0.68  (hard to translate)
   - Δde = 0.53 ✓ (improvement!)

2. Constraint Score:
   - Length: 11 tokens (within bounds) ✓
   - Lexical diversity: 0.91 (11 unique / 11 total) ✓
   - constraint(edit) = 0.72

3. Verifier Score:
   - LM perplexity: 24.3
   - verify(edit) = 0.85 (grammatical) ✓

4. Combined:
   - L = 1 - (1.0*0.53 + 0.3*0.72 + 0.3*0.85) / 1.6
   - L = 0.12 (low loss = good)
   - Reward = -0.12 → Policy learns to maximize this!
```

---

## Memory Optimizations

The PPO implementation includes extensive memory optimizations for training on consumer GPUs:

### Automatic Features

1. **Mixed Precision Training (FP16)**
   - Automatically enabled on CUDA devices
   - Reduces model memory by ~50%
   - Maintains FP32 gradients for stability

2. **Gradient Checkpointing**
   - Enabled for the policy model
   - Trades compute for memory during backprop
   - Reduces activation memory by ~40%

3. **Mini-batch Processing**
   - PPO processes 1 sample at a time during updates
   - Prevents OOM during gradient computation

4. **Periodic Cache Clearing**
   - CUDA cache cleared after generation and PPO steps
   - Prevents memory fragmentation over long runs

### Manual Optimizations

**Offload models to CPU:**
```bash
python -m src.RL_ppo_training \
  --mt_device cpu \              # Save ~300MB GPU
  --lm_verifier_device cpu \     # Save ~250MB GPU
  --comet_accelerator cpu        # Save ~2GB GPU
```

**Reduce batch size:**
```bash
--batch_size 1    # Minimum: 1 sample per step
```

**Reduce generation length:**
```bash
--gen_max_new_tokens 32    # Shorter = less memory
```

**Use smaller model:**
```bash
--base_model gpt2    # ~500MB vs 1.2GB for Qwen3
```

### Memory Budget Breakdown (Qwen3-0.6B on GTX 1080 Ti)

| Component | VRAM (FP16) | Configurable |
|-----------|-------------|--------------|
| Policy model | ~600MB | Base model choice |
| Reference model | ~600MB | Base model choice |
| Value head | ~50MB | - |
| Optimizer states | ~1.2GB | - |
| Activations/gradients | ~2-3GB | Batch size, seq length |
| MT translator | ~300MB | `--mt_device cpu` |
| COMET QE | ~2GB | `--comet_accelerator cpu` |
| LM Verifier | ~250MB | `--lm_verifier_device cpu` |
| Generation buffer | ~500MB | Batch size, max tokens |
| **Total** | **~10-11GB** | **Can reduce to ~6GB** |

---

## Weights & Biases Integration

The training script logs comprehensive metrics to Weights & Biases:

### Logged Metrics

**Reward Metrics:**
- `reward/mean`: Average reward across batch
- `reward/min`: Minimum reward in batch
- `reward/max`: Maximum reward in batch
- `reward/constraint_mean`: Average constraint score
- `reward/verifier_mean`: Average verifier score
- `reward/difficulty_delta_mean`: Average difficulty improvement

**PPO Statistics:**
- `ppo/loss/total`: Total PPO loss
- `objective/kl`: KL divergence from reference policy
- `objective/entropy`: Policy entropy (exploration)
- `ppo/policy/approxkl`: Approximate KL divergence
- `ppo/policy/clipfrac`: Fraction of clipped updates
- `ppo/returns/mean`: Mean returns
- `ppo/val/mean`: Mean value estimates

**Generation Statistics:**
- `generation/mean_response_length`: Average tokens generated
- `generation/sample_text`: Sample output (HTML preview)

**System Metrics:**
- Learning rate
- GPU memory usage
- Training step time

### Configuration

```bash
# Enable WandB (default)
python -m src.RL_ppo_training --wandb_project my-project --wandb_run_name exp-1

# Disable WandB
python -m src.RL_ppo_training --no_wandb
```

### Artifact Logging

At the end of training, the script logs:
- Final model checkpoint as WandB artifact
- Final sample generation
- Full hyperparameter config

---

## SLURM Cluster Deployment

The repository includes production-ready SLURM scripts in the [scripts/](scripts/) directory:

```bash
#!/bin/bash
#SBATCH --account=deep_learning
#SBATCH --output=logs/mtbreaker_%j.out
#SBATCH --job-name=setup-RL
#SBATCH --gpus 1080ti:1

. /etc/profile.d/modules.sh
module add cuda/12.6

source "$HOME/venvs/mtbreaker/bin/activate"
cd "$HOME/DeepLearning/"
mkdir -p logs

python -m src.RL_ppo_training \
  --seeds data/seeds.txt \
  --k 200 \
  --steps 300 \
  --batch_size 2 \
  --temperature 0.4
```

**Submit job:**
```bash
sbatch scripts/run.sh
```

**Monitor logs:**
```bash
tail -f logs/mtbreaker_<job_id>.out
```

---

## Model Inference After Training

### Loading the Trained Policy

```python
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead

# Load checkpoint
model_path = "checkpoints/gpt2-ppo-method2"
base_model = "Qwen/Qwen3-0.6B-Base"  # Must match training

tokenizer = AutoTokenizer.from_pretrained(base_model)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLMWithValueHead.from_pretrained(model_path)
model.eval()

# Move to device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
```

### Generating Difficult Sentences

```python
seed = "The cat sat on the mat."
instruction = f'Rewrite this sentence to be extremely difficult for machine translation using idioms, ambiguity, and wordplay, while keeping it grammatically correct English: "{seed}"\n\nOnly return the single edited sentence.'

# Tokenize
inputs = tokenizer(instruction, return_tensors="pt").to(device)

# Generate
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=True,
        top_p=0.8,
        temperature=0.7,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )

# Decode (only the generated part, excluding prompt)
input_len = inputs.input_ids.shape[1]
generated_ids = outputs[0][input_len:]
generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

print(f"Seed: {seed}")
print(f"Generated: {generated_text}")
```

### Batch Inference

```python
seeds = [
    "The weather is nice.",
    "He went to the bank.",
    "She saw the bat."
]

prompts = [
    f'Rewrite this sentence to be extremely difficult for machine translation using idioms, ambiguity, and wordplay, while keeping it grammatically correct English: "{s}"\n\nOnly return the single edited sentence.'
    for s in seeds
]

# Batch tokenization
inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)

# Batch generation
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=True,
        top_p=0.8,
        temperature=0.7,
        num_return_sequences=1
    )

# Decode all
for i, (seed, output) in enumerate(zip(seeds, outputs)):
    input_len = len(tokenizer.encode(prompts[i]))
    generated = tokenizer.decode(output[input_len:], skip_special_tokens=True)
    print(f"{i+1}. Seed: {seed}")
    print(f"   Generated: {generated}\n")
```

---

## Optional: Analysis Pipeline (method2.py)

While PPO training doesn't require it, you can use `method2.py` to analyze the reward function offline:

```bash
python -m src.method2 --seeds data/seeds.txt --k 100
```

This generates `data/method2_results.jsonl` with detailed metrics for each seed:

```json
{
  "orig": "He saw her duck.",
  "edit": "He observed the bird dive beneath the bridge.",
  "de_old": 0.42,
  "de_new": 0.63,
  "delta_de": 0.21,
  "constraint": 0.37,
  "verify": 0.85,
  "L": 0.43
}
```

Use this to:
- Understand what makes sentences hard to translate
- Tune reward weights (x, y, z)
- Identify promising seed sentences
- Debug reward computation

---

## Hardware Requirements & Performance

### Minimum Requirements

**PPO Training:**
- GPU: 11GB VRAM (GTX 1080 Ti, RTX 2080 Ti, RTX 3080)
- RAM: 32GB recommended
- Storage: 20GB for models, checkpoints, and cache

**Memory-Optimized Setup:**
- GPU: 6-8GB VRAM (GTX 1060, RTX 2060)
- Use CPU offloading: `--mt_device cpu --lm_verifier_device cpu --comet_accelerator cpu`
- Use smaller model: `--base_model gpt2`
- Reduce batch size: `--batch_size 1`

### Performance Benchmarks

**Training Speed (GTX 1080 Ti):**
- Qwen3-0.6B, batch_size=2: ~200 steps in 2-3 hours
- GPT-2, batch_size=2: ~200 steps in 1.5-2 hours
- Time per step: ~30-45 seconds (includes generation + reward computation + PPO update)

**Convergence:**
- Meaningful improvements: ~50-100 steps
- Strong performance: ~200-300 steps
- Diminishing returns: >500 steps

**Throughput:**
- ~4-5 edits/minute (batch_size=2, Qwen3)
- ~6-8 edits/minute (batch_size=2, GPT-2)

---

## Troubleshooting

### Common Issues

**Out of Memory (OOM) Errors:**
```bash
# Solution 1: Reduce batch size
--batch_size 1

# Solution 2: Offload models to CPU
--mt_device cpu --lm_verifier_device cpu --comet_accelerator cpu

# Solution 3: Reduce generation length
--gen_max_new_tokens 32

# Solution 4: Use smaller base model
--base_model gpt2
```

**COMET QE fails to load:**
```bash
# Force CPU mode
--comet_accelerator cpu

# Check installation
pip install unbabel-comet==2.2.2

# Check internet (first run downloads model)
```

**Empty or repetitive generations:**
```bash
# Increase temperature
--temperature 0.9

# Adjust top_p
--top_p 0.95

# Check model loading with test script
python test_model_inference.py
```

**WandB errors:**
```bash
# Disable WandB
--no_wandb

# Or login first
wandb login
```

**CUDA out of memory during generation:**
```bash
# The model generates full sequences in memory
# Reduce max tokens or batch size
--gen_max_new_tokens 48 --batch_size 1
```

### Testing & Debugging Scripts

**Check tokenizer:**
```bash
python test_scripts/check_tokenizer.py
```

**Test model inference:**
```bash
python test_scripts/test_model_inference.py
```

**Verify generation:**
```bash
bash scripts/verify.sh
# or
python test_scripts/verify_gpt2_generation.py
```

**Test on SLURM:**
```bash
bash scripts/test_model.sh
```

---

## Dependencies

### Core Requirements

**Deep Learning Framework:**
- Python 3.10
- PyTorch 2.9.1+cu126 (CUDA 12.6)
- Transformers 4.57.1 (HuggingFace models and tokenizers)
- TRL 0.8.6 (PPO implementation)
- Accelerate 0.34.2 (distributed training, mixed precision)
- PEFT 0.12.0 (parameter-efficient fine-tuning utilities)

**Evaluation & Scoring:**
- Unbabel-COMET 2.2.2 (COMET QE for translation quality)
- SentencePiece 0.1.99 (tokenization)
- Sacremoses 0.1.1 (text normalization)
- SacreBLEU 2.5.1 (evaluation metrics)

**Data Processing:**
- Datasets 2.21.0 (HuggingFace datasets)
- NumPy 1.26.4
- Pandas 2.3.3
- PyArrow 22.0.0

**Experiment Tracking:**
- Weights & Biases 0.23.0 (optional, for logging)
- TensorBoard (via PyTorch Lightning 2.5.6)

**Additional Libraries:**
- tqdm 4.67.1 (progress bars)
- PyYAML 6.0.3 (configuration files)
- Jinja2 3.1.6 (templating)
- Pillow 11.3.0 (image processing)
- Requests 2.32.5 (HTTP library)

### NVIDIA CUDA Libraries (Included with PyTorch)

- nvidia-cuda-runtime-cu12 12.6.77
- nvidia-cudnn-cu12 9.10.2.21
- nvidia-cublas-cu12 12.6.4.1
- nvidia-cufft-cu12 11.3.0.4
- nvidia-cusolver-cu12 11.7.1.2
- nvidia-cusparse-cu12 12.5.4.2
- nvidia-nccl-cu12 2.27.5

### Installation

**Option 1: Using Conda (Recommended)**
```bash
conda env create -f env.yaml
conda activate mtbreaker
```

**Option 2: Using pip with requirements.txt**
```bash
# Create virtual environment
python3.10 -m venv venvs/mtbreaker
source venvs/mtbreaker/bin/activate  # On Windows: venvs\mtbreaker\Scripts\activate

# Install PyTorch with CUDA 12.6 first
pip install torch==2.9.1+cu126 torchvision==0.24.1+cu126 torchaudio==2.9.1+cu126 \
  --index-url https://download.pytorch.org/whl/cu126

# Install all other dependencies
pip install -r requirements.txt
```

**Option 3: Manual pip installation**
```bash
# Create virtual environment
python3.10 -m venv venvs/mtbreaker
source venvs/mtbreaker/bin/activate

# Install PyTorch with CUDA 12.6
pip install torch==2.9.1+cu126 torchvision==0.24.1+cu126 torchaudio==2.9.1+cu126 \
  --index-url https://download.pytorch.org/whl/cu126

# Install core dependencies
pip install transformers==4.57.1 \
  trl==0.8.6 \
  accelerate==0.34.2 \
  peft==0.12.0 \
  datasets==2.21.0

# Install evaluation libraries
pip install unbabel-comet==2.2.2 \
  sentencepiece==0.1.99 \
  sacremoses==0.1.1 \
  sacrebleu==2.5.1

# Install utilities
pip install numpy==1.26.4 \
  pandas==2.3.3 \
  tqdm==4.67.1 \
  pyyaml==6.0.3

# Optional: Install Weights & Biases for experiment tracking
pip install wandb==0.23.0
```

**Verify Installation:**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "from comet import download_model; print('COMET: OK')"
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{breaking-mt-2024,
  title={Breaking-MT: Reinforcement Learning Framework for Adversarial Machine Translation},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/breaking-mt}
}
```

---

## License

This project is released under the MIT License.

---
