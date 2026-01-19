from unsloth import FastLanguageModel
import argparse
import os
import re
import math
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import torch
from sentinel_metric import download_model as sentinel_download_model
from sentinel_metric import load_from_checkpoint as sentinel_load_from_checkpoint
from datasets import load_dataset, Dataset

from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from transformers import TrainerCallback

from trl import GRPOConfig, GRPOTrainer

import wandb



if os.path.exists("wandb.key"):
    print("Found wandb.key, logging in...")
    with open("wandb.key") as f:
        wandb.login(key=f.read().strip())
else:
    print("No wandb.key found")



SYSTEM_PROMPT = """You are a rewriting assistant.
Rewrite the given English sentence so that it becomes harder to translate into other languages,
while preserving the original meaning as much as possible.

Guidelines:
- Keep the rewrite fluent, grammatical English (no gibberish).
- You may increase translation difficulty by using idioms, phrasal verbs, wordplay, ambiguity,
  nested clauses, unusual but natural collocations, or culturally-bound expressions.
- Do NOT add explanations, commentary, or multiple options.
- Keep length roughly similar: between half and twice the original length.
- Keep names, numbers, dates, and proper nouns unchanged unless absolutely necessary.

Output EXACTLY in this format, and only this format:
<rewrite>
...
</rewrite>

Important:
- The output must start with <rewrite> and end with </rewrite>.
"""

USER_TEMPLATE = """Original sentence:
{sentence}

Rewrite it to be harder to translate.

Output:
<rewrite>
"""

def make_prompt(sentence: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(sentence=sentence)},
    ]



class Sentinel:
    def __init__(self, model_id: str = "Prosho/sentinel-src-25", device: str = "cuda:0", batch_size: int = 8):
        self.model_id = model_id
        self.device = device
        self.batch_size = batch_size

        print(f"[Sentinel] Loading {model_id} ...")
        ckpt = sentinel_download_model(model_id)
        self.model = sentinel_load_from_checkpoint(ckpt)

        self._gpus = 1 if device.startswith("cuda") and torch.cuda.is_available() else 0
        print(f"[Sentinel] Ready (batch_size={batch_size}, gpus={self._gpus}).")

    @torch.inference_mode()
    def difficulty(self, sentences: List[str]) -> List[float]:
        data = [{"src": s} for s in sentences]
        out = self.model.predict(data, batch_size=self.batch_size, gpus=self._gpus)
        return list(map(float, out.scores))



# Embedding similarity
class Embedder:
    def __init__(self, model_name: str, device: str):
        self.device = device
        print(f"[Embedder] Loading {model_name} on {device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

    @torch.inference_mode()
    def encode(self, texts: List[str], max_length: int = 256) -> torch.Tensor:
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(self.device)
        out = self.model(**batch)
        last = out.last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1)
        pooled = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
        return pooled

    @torch.inference_mode()
    def cosine_sim(self, a: List[str], b: List[str]) -> List[float]:
        ea = self.encode(a)
        eb = self.encode(b)
        sims = (ea * eb).sum(dim=-1)
        return sims.detach().float().cpu().tolist()



# CoLA acceptability
class AcceptabilityClassifier:
    def __init__(self, model_name: str, device: str):
        self.device = device
        print(f"[CoLA] Loading {model_name} on {device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        self.model.eval()

        self.accept_label_id = 1
        if hasattr(self.model.config, "id2label") and isinstance(self.model.config.id2label, dict):
            for k, v in self.model.config.id2label.items():
                vv = str(v).lower()
                if "acceptable" in vv or "grammatical" in vv or vv in ("accept", "ok", "yes"):
                    self.accept_label_id = int(k)
                    break

    @torch.inference_mode()
    def p_acceptable(self, texts: List[str], max_length: int = 256) -> List[float]:
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(self.device)
        logits = self.model(**batch).logits
        probs = torch.softmax(logits, dim=-1)
        p = probs[:, self.accept_label_id]
        return p.detach().float().cpu().tolist()



# Extraction + heuristics
REWRITE_SOFT_RE = re.compile(r"<rewrite>\s*(.*?)\s*</rewrite>", re.DOTALL)
REWRITE_STRICT_RE = re.compile(r"^<rewrite>\n([^\n]+)\n</rewrite>\n?$")

def extract_rewrite(text: str) -> str:
    # 1)Ideal case: <rewrite> ... </rewrite>
    m = REWRITE_SOFT_RE.search(text)
    if m:
        s = re.sub(r"\s+", " ", m.group(1).strip())
        return s

    t = (text or "").strip()

    # 2) Common failure: starts with </rewrite> then the sentence
    t = re.sub(r"^\s*</rewrite>\s*", "", t)

    # 3) Has opening tag but missing/garbled closing tag
    if "<rewrite>" in t:
        after = t.split("<rewrite>", 1)[1].strip()
        after = after.split("</rewrite>", 1)[0].strip()
        after = re.sub(r"\s+", " ", after)
        return after

    # 4) No tags at all: fall back to first non-empty line
    first_line = next((ln.strip() for ln in t.splitlines() if ln.strip()), "")
    first_line = re.sub(r"\s+", " ", first_line)
    return first_line

def is_strict_format(text: str) -> bool:
    return REWRITE_STRICT_RE.match(text) is not None

def word_count(s: str) -> int:
    return len(re.findall(r"\b\w+\b", s))

def sentence_count_heuristic(s: str) -> int:
    s2 = re.sub(r"\b(e\.g|i\.e|mr|mrs|dr)\.", r"\1", s, flags=re.IGNORECASE)
    return len(re.findall(r"[.!?]", s2))

def repetition_penalty_scalar(s: str) -> float:
    if not s:
        return 0.0
    if re.search(r"(.)\1\1\1\1", s):
        return 1.0
    toks = re.findall(r"\b\w+\b", s.lower())
    if len(toks) >= 12:
        unique = len(set(toks))
        if unique / len(toks) < 0.5:
            return 0.5
    return 0.0



# Reward weights
@dataclass
class RewardWeights:
    diff: float = 1.0
    format_strict: float = 0.4
    format_soft: float = 0.1
    length: float = 0.2
    similarity: float = 0.4
    acceptability: float = 0.6
    single_sentence: float = 0.05
    repetition_pen: float = 0.6



# W&B rollout logging (training rollouts)
CURRENT_STEP = [0]
RAW_DELTA_ACCUM = {"sum": 0.0, "count": 0}

class RolloutWandbLogger:
    """Buffer rows (dicts) and flush to explained W&B Table."""
    def __init__(self, enabled: bool, samples_per_flush: int = 16, max_buffer: int = 512):
        self.enabled = enabled and (wandb is not None)
        self.samples_per_flush = samples_per_flush
        self.max_buffer = max_buffer
        self._buf: List[Dict[str, Any]] = []

    def add_many(self, rows: List[Dict[str, Any]]) -> None:
        if not self.enabled or not rows:
            return
        self._buf.extend(rows)
        if len(self._buf) > self.max_buffer:
            self._buf = self._buf[-self.max_buffer:]

    def flush(self, step: int) -> None:
        if not self.enabled or len(self._buf) == 0:
            return
        take = min(len(self._buf), self.samples_per_flush)
        rows = self._buf[:take]
        self._buf = self._buf[take:]

        columns = [
            "step", "original", "rewrite", "baseline_sentinel", "rewrite_sentinel", "raw_delta",
            "r_diff", "r_format_strict", "r_format_soft", "r_tag_order", "r_len", "r_sim", "r_acc", "r_single", "r_rep_pen",
            "r_total", "raw_completion",
        ]
        data = [[r.get(c) for c in columns] for r in rows]
        table = wandb.Table(columns=columns, data=data)
        wandb.log({"train_rollouts": table})

class StepSyncCallback(TrainerCallback):
    def __init__(self, rollout_logger, flush_every=10):
        self.rollout_logger = rollout_logger
        self.flush_every = flush_every

    def on_step_end(self, args, state, control, **kwargs):
        CURRENT_STEP[0] = int(state.global_step)

        trainer = kwargs.get("trainer", None)
        is_main = True
        if trainer is not None and hasattr(trainer, "accelerator"):
            is_main = bool(trainer.accelerator.is_main_process)

        if is_main:
            #log mean raw_delta
            if RAW_DELTA_ACCUM["count"] > 0:
                mean_raw_delta = RAW_DELTA_ACCUM["sum"] / RAW_DELTA_ACCUM["count"]
                wandb.log({
                    "reward/raw_delta_mean": mean_raw_delta,
                    "reward/raw_delta_count": RAW_DELTA_ACCUM["count"],
                })

                RAW_DELTA_ACCUM["sum"] = 0.0
                RAW_DELTA_ACCUM["count"] = 0

            if state.global_step > 0 and state.global_step % self.flush_every == 0:
                self.rollout_logger.flush(step=int(state.global_step))


# combined reward function
def build_total_reward_func(
    sentinel: Sentinel,
    embedder: Embedder,
    accept_clf: AcceptabilityClassifier,
    weights: RewardWeights,
    rollout_logger: Optional[RolloutWandbLogger],
    sim_floor: float = 0.78,
    diff_clip: float = 2.0,
    log_random_k: int = 4,
):
    """
    Returns ONE TRL reward function that:
    - computes all reward components in one pass
    - returns total reward
    """

    def reward_func(completions, baseline_difficulty, original_text, **kwargs) -> List[float]:
        responses = [c[0]["content"] for c in completions]
        rewrites = [extract_rewrite(r) for r in responses]


        r_format_strict = [weights.format_strict if is_strict_format(r) else 0.0 for r in responses]
        r_format_soft = [weights.format_soft if REWRITE_SOFT_RE.search(r) else 0.0 for r in responses]

        has_rw = [bool(rw) for rw in rewrites]
        idxs = [i for i, ok in enumerate(has_rw) if ok]

        # Tag order penalty: discourage outputs that contain </rewrite> before <rewrite>
        r_tag_order_pen = [0.0] * len(responses)
        for i, resp in enumerate(responses):
            a = resp.find("<rewrite>")
            b = resp.find("</rewrite>")
            if b != -1 and (a == -1 or b < a):
                r_tag_order_pen[i] = -0.2  # small but persistent penalty

        # Sentinel scores (rewrite only)
        rewrite_scores = [0.0] * len(rewrites)
        if idxs:
            scs = sentinel.difficulty([rewrites[i] for i in idxs])
            for i, sc in zip(idxs, scs):
                rewrite_scores[i] = float(sc)

        # Similarity
        sims = [0.0] * len(rewrites)
        if idxs:
            ss = embedder.cosine_sim([original_text[i] for i in idxs], [rewrites[i] for i in idxs])
            for i, s in zip(idxs, ss):
                sims[i] = float(s)

        # Acceptability
        p_acc = [0.0] * len(rewrites)
        if idxs:
            ps = accept_clf.p_acceptable([rewrites[i] for i in idxs])
            for i, p in zip(idxs, ps):
                p_acc[i] = float(p)

        r_diff, r_len, r_sim, r_acc, r_single, r_rep_pen = [], [], [], [], [], []
        raw_deltas = []

        for i in range(len(rewrites)):
            orig = original_text[i]
            base = float(baseline_difficulty[i])
            rw = rewrites[i]

            if not rw:
                r_diff.append(-weights.diff)
                r_len.append(-weights.length)
                r_sim.append(-weights.similarity)
                r_acc.append(-weights.acceptability)
                r_single.append(-weights.single_sentence)
                r_rep_pen.append(0.0)
                raw_deltas.append(0.0)
                print("XXXXXXXXXXXXXXXXXX")
                continue

            raw_delta = base - float(rewrite_scores[i])
            raw_deltas.append(raw_delta)
            RAW_DELTA_ACCUM["sum"] += raw_delta
            RAW_DELTA_ACCUM["count"] += 1

            raw_delta = raw_delta * 5
            d = max(-diff_clip, min(diff_clip, raw_delta))
            r_diff.append(weights.diff * d)

            # length shaping
            ow = max(1, word_count(orig))
            rwc = max(1, word_count(rw))
            ratio = rwc / ow
            if ratio < 0.5 or ratio > 2.0:
                r_len.append(-weights.length)
            else:
                dist = abs(math.log(ratio))
                shaped = max(0.0, 1.0 - dist / math.log(2.0))
                r_len.append(weights.length * shaped)

            # similarity
            sim = sims[i]
            if sim < sim_floor:
                r_sim.append(-weights.similarity)
            else:
                shaped = (sim - sim_floor) / max(1e-6, (1.0 - sim_floor))
                r_sim.append(weights.similarity * shaped)

            # acceptability
            shaped = (p_acc[i] - 0.5) * 2.0
            r_acc.append(weights.acceptability * shaped)

            # single sentence
            n = sentence_count_heuristic(rw)
            r_single.append(weights.single_sentence if n <= 1 else 0.0)

            # repetition penalty
            rep = repetition_penalty_scalar(rw)
            r_rep_pen.append(-weights.repetition_pen * rep)

        total = []
        for i in range(len(rewrites)):
            total_i = (
                r_diff[i]
                + r_format_strict[i]
                + r_format_soft[i]
                + r_len[i]
                + r_tag_order_pen[i]
                + r_sim[i]
                + r_acc[i]
                + r_single[i]
                + r_rep_pen[i]
            )
            total.append(float(total_i))

        # Enqueue some training rollouts for W&B
        if rollout_logger is not None and rollout_logger.enabled:
            trainer = kwargs.get("trainer", None)
            is_main = True
            if trainer is not None and hasattr(trainer, "accelerator"):
                is_main = bool(trainer.accelerator.is_main_process)

            if is_main:
                candidates = list(range(len(rewrites)))
                # prefer those with a rewrite extracted
                candidates.sort(key=lambda i: 0 if has_rw[i] else 1)
                chosen = candidates[:min(len(candidates), log_random_k)]
                # add a bit of randomness when plenty of good candidates
                if sum(has_rw) > log_random_k:
                    good = [i for i in candidates if has_rw[i]]
                    chosen = random.sample(good, log_random_k)

                step = int(CURRENT_STEP[0])
                rows = []
                for i in chosen:
                    rows.append({
                        "step": step,
                        "original": original_text[i],
                        "rewrite": rewrites[i],
                        "baseline_sentinel": float(baseline_difficulty[i]),
                        "rewrite_sentinel": float(rewrite_scores[i]),
                        "raw_delta": float(raw_deltas[i]),
                        "r_diff": float(r_diff[i]),
                        "r_format_strict": float(r_format_strict[i]),
                        "r_format_soft": float(r_format_soft[i]),
                        "r_tag_order": float(r_tag_order_pen[i]),
                        "r_len": float(r_len[i]),
                        "r_sim": float(r_sim[i]),
                        "r_acc": float(r_acc[i]),
                        "r_single": float(r_single[i]),
                        "r_rep_pen": float(r_rep_pen[i]),
                        "r_total": float(total[i]),
                        "raw_completion": responses[i][:2000],
                    })
                rollout_logger.add_many(rows)

        return total

    return reward_func


def load_or_build_dataset(dataset_name: str, split: str, max_chars: int) -> Dataset:
    print(f"[Dataset] Loading {dataset_name} ({split}) ...")
    ds = load_dataset(dataset_name, split=split)
    cols = set(ds.column_names)

    if "text" in cols and "sentinel_score" in cols:
        ds = ds.filter(lambda ex: len(ex["text"]) < max_chars)
        ds = ds.map(lambda ex: {
            "prompt": make_prompt(ex["text"]),
            "original_text": ex["text"],
            "baseline_difficulty": float(ex["sentinel_score"]),
        })
        return ds

    if "original_text" in cols and "baseline_difficulty" in cols:
        ds = ds.filter(lambda ex: len(ex["original_text"]) < max_chars)
        ds = ds.map(lambda ex: {
            "prompt": make_prompt(ex["original_text"]),
            "original_text": ex["original_text"],
            "baseline_difficulty": float(ex["baseline_difficulty"]),
        })
        return ds



def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--dataset", type=str, default="matafix/english-sentences-with-sentinel")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--max_chars", type=int, default=355)

    p.add_argument("--base_model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--output_dir", type=str, default="./grpo_harder_translate_lora")

    # Unsloth/LoRA
    p.add_argument("--max_seq_length", type=int, default=1024)
    p.add_argument("--load_in_4bit", action="store_true", default=True)


    p.add_argument("--fast_inference", action="store_true", default=False)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    p.add_argument("--max_lora_rank", type=int, default=32)

    p.add_argument("--lora_rank", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")

    # GRPO
    p.add_argument("--learning_rate", type=float, default=5e-6)
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--max_prompt_length", type=int, default=512)
    p.add_argument("--max_completion_length", type=int, default=128)
    p.add_argument("--num_generations", type=int, default=4)
    p.add_argument("--logging_steps", type=int, default=1)
    p.add_argument("--save_steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)

    # Reward models
    p.add_argument("--reward_device", type=str, default="cuda:0")
    p.add_argument("--sentinel_model", type=str, default="Prosho/sentinel-src-25")
    p.add_argument("--sentinel_batch_size", type=int, default=8)

    p.add_argument("--embedder_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--cola_model", type=str, default="textattack/roberta-base-CoLA")

    p.add_argument("--sim_floor", type=float, default=0.78)
    p.add_argument("--diff_clip", type=float, default=2.0)

    # Reward weights
    p.add_argument("--w_diff", type=float, default=1.0)
    p.add_argument("--w_format_strict", type=float, default=0.4)
    p.add_argument("--w_format_soft", type=float, default=0.1)
    p.add_argument("--w_length", type=float, default=0.4)
    p.add_argument("--w_similarity", type=float, default=0.4)
    p.add_argument("--w_acceptability", type=float, default=0.6)
    p.add_argument("--w_single_sentence", type=float, default=0.05)
    p.add_argument("--w_repetition_pen", type=float, default=0.6)

    # W&B
    p.add_argument("--wandb", action="store_true", default=True)
    p.add_argument("--wandb_project", type=str, default="mt-breaker")
    p.add_argument("--wandb_run_name", type=str, default="grpo-train-rollouts")
    p.add_argument("--wandb_flush_every", type=int, default=10)
    p.add_argument("--wandb_samples_per_flush", type=int, default=16)

    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    rollout_logger = RolloutWandbLogger(
        enabled=args.wandb,
        samples_per_flush=args.wandb_samples_per_flush,
        max_buffer=512,
    )

    dataset = load_or_build_dataset(args.dataset, args.split, args.max_chars)

    print(f"[Model] Loading {args.base_model} ...")
    fp_kwargs = dict(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
    )
    if args.fast_inference:
        fp_kwargs.update(dict(
            fast_inference=True,
            max_lora_rank=args.max_lora_rank,
            gpu_memory_utilization=args.gpu_memory_utilization,
        ))

    model, tokenizer = FastLanguageModel.from_pretrained(**fp_kwargs)

    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=target_modules,
        lora_alpha=args.lora_rank,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    reward_device = args.reward_device
    if reward_device.startswith("cuda") and not torch.cuda.is_available():
        reward_device = "cpu"

    sentinel = Sentinel(model_id=args.sentinel_model, device=reward_device, batch_size=args.sentinel_batch_size)
    embedder = Embedder(model_name=args.embedder_model, device=reward_device)
    accept_clf = AcceptabilityClassifier(model_name=args.cola_model, device=reward_device)

    weights = RewardWeights(
        diff=args.w_diff,
        format_strict=args.w_format_strict,
        format_soft=args.w_format_soft,
        length=args.w_length,
        similarity=args.w_similarity,
        acceptability=args.w_acceptability,
        single_sentence=args.w_single_sentence,
        repetition_pen=args.w_repetition_pen,
    )

    total_reward_func = build_total_reward_func(
        sentinel=sentinel,
        embedder=embedder,
        accept_clf=accept_clf,
        weights=weights,
        rollout_logger=rollout_logger,
        sim_floor=args.sim_floor,
        diff_clip=args.diff_clip,
        log_random_k=4,
    )

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        seed=args.seed,
        report_to=(["wandb"] if args.wandb else ["none"]),
        run_name=(args.wandb_run_name if args.wandb else None),
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[total_reward_func],
        args=training_args,
        train_dataset=dataset,
    )

    trainer.add_callback(StepSyncCallback(rollout_logger, flush_every=args.wandb_flush_every))

    trainer.train()

    rollout_logger.flush(step=int(CURRENT_STEP[0]))

    print(f"[Save] Saving adapter to {args.output_dir} ...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("[Done]")

if __name__ == "__main__":
    main()
