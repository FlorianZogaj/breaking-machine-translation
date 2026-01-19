
"""
Evaluation script for GRPO "harder to translate" rewrites.

Compares:
- mean baseline Sentinel score (original sentences)
- mean Sentinel score of rewrites from the unfinetuned base model
- mean Sentinel score of rewrites from the finetuned model

Only counts samples where outputs are valid (constraints met):
- length ratio: 0.5x .. 2.0x (word-based)
- semantic similarity >= threshold (MiniLM cosine)
- grammatical acceptability >= threshold (CoLA P(acceptable))
- rewrite extraction succeeded (non-empty)
- optional: single sentence heuristic

Uses the INTERSECTION of valid samples for base and finetuned (fair comparison).
Also reports validity rates for each model.
"""

from unsloth import FastLanguageModel
import argparse
import re
import math
from typing import List, Dict, Tuple
from transformers import set_seed
import torch
from datasets import load_dataset
from datetime import datetime
import os
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification



from sentinel_metric import download_model as sentinel_download_model
from sentinel_metric import load_from_checkpoint as sentinel_load_from_checkpoint



# Prompting (matches training)
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
YOUR_SINGLE_SENTENCE_REWRITE
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

REWRITE_SOFT_RE = re.compile(r"<rewrite>\s*(.*?)\s*</rewrite>", re.DOTALL)

def make_prompt(sentence: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(sentence=sentence)},
    ]



def extract_rewrite(text: str) -> str:
    m = REWRITE_SOFT_RE.search(text)
    if m:
        return re.sub(r"\s+", " ", m.group(1).strip())

    t = (text or "").strip()
    t = re.sub(r"^\s*</rewrite>\s*", "", t)

    if "<rewrite>" in t:
        after = t.split("<rewrite>", 1)[1].strip()
        after = after.split("</rewrite>", 1)[0].strip()
        return re.sub(r"\s+", " ", after)

    first_line = next((ln.strip() for ln in t.splitlines() if ln.strip()), "")
    return re.sub(r"\s+", " ", first_line)

def word_count(s: str) -> int:
    return len(re.findall(r"\b\w+\b", s))

def sentence_count_heuristic(s: str) -> int:
    s2 = re.sub(r"\b(e\.g|i\.e|mr|mrs|dr)\.", r"\1", s, flags=re.IGNORECASE)
    return len(re.findall(r"[.!?]", s2))

def set_all_seeds(seed: int):
    set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def mean_pairwise_cosine_similarity(emb: torch.Tensor) -> float:
    """
    emb: (N, D) L2-normalized embeddings
    """
    n = emb.shape[0]
    if n < 2:
        return float("nan")
    sim = emb @ emb.T  # cosine similarity
    triu = torch.triu(sim, diagonal=1)
    return (triu.sum() / (n * (n - 1) / 2)).item()



# Evaluators
class Sentinel:
    def __init__(self, model_id: str = "Prosho/sentinel-src-25", device: str = "cuda:0", batch_size: int = 16):
        print(f"[Sentinel] Loading {model_id} ...")
        ckpt = sentinel_download_model(model_id)
        self.model = sentinel_load_from_checkpoint(ckpt)
        self.batch_size = batch_size
        self._gpus = 1 if device.startswith("cuda") and torch.cuda.is_available() else 0

    @torch.inference_mode()
    def difficulty(self, sentences: List[str]) -> List[float]:
        clean = [s if s.strip() else "EMPTY" for s in sentences]
        data = [{"src": s} for s in clean]
        out = self.model.predict(data, batch_size=self.batch_size, gpus=self._gpus)
        return list(map(float, out.scores))

class Embedder:
    def __init__(self, model_name: str, device: str):
        print(f"[Embedder] Loading {model_name} on {device} ...")
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

    @torch.inference_mode()
    def encode(self, texts: List[str], max_length: int = 256) -> torch.Tensor:
        batch = self.tok(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(self.device)
        out = self.model(**batch)
        last = out.last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1)
        pooled = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return torch.nn.functional.normalize(pooled, p=2, dim=-1)

    @torch.inference_mode()
    def cosine_sim(self, a: List[str], b: List[str]) -> List[float]:
        ea = self.encode(a)
        eb = self.encode(b)
        return (ea * eb).sum(dim=-1).detach().float().cpu().tolist()

class AcceptabilityClassifier:
    def __init__(self, model_name: str, device: str):
        print(f"[CoLA] Loading {model_name} on {device} ...")
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_name)
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
        batch = self.tok(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(self.device)
        logits = self.model(**batch).logits
        probs = torch.softmax(logits, dim=-1)
        return probs[:, self.accept_label_id].detach().float().cpu().tolist()



# Generation
def build_prompt_texts(tokenizer, originals: List[str]) -> List[str]:
    msgs = [make_prompt(s) for s in originals]
    return [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]

@torch.inference_mode()
def generate_rewrites(
    model,
    tokenizer,
    originals: List[str],
    device: str,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Tuple[List[str], List[str]]:
    FastLanguageModel.for_inference(model)

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt_texts = build_prompt_texts(tokenizer, originals)

    rewrites, raw_completions = [], []
    for i in range(0, len(originals), batch_size):
        batch_prompts = prompt_texts[i:i+batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)

        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False, # deterministic generation for evaluation
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
        )

        input_len = inputs["input_ids"].shape[1]
        gen_tokens = out[:, input_len:]
        decoded = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)

        for d in decoded:
            comp = d.strip()
            raw_completions.append(comp)
            rewrites.append(extract_rewrite(comp))

    return rewrites, raw_completions



# Validity filtering
def compute_valid_mask(
    originals: List[str],
    rewrites: List[str],
    sims: List[float],
    acc_ps: List[float],
    sim_floor: float,
    acc_floor: float,
    enforce_single_sentence: bool,
) -> Tuple[List[bool], List[str]]:
    mask, reasons = [], []
    for o, r, sim, p in zip(originals, rewrites, sims, acc_ps):
        if not r.strip():
            mask.append(False); reasons.append("empty_rewrite"); continue

        ow = max(1, word_count(o))
        rw = max(1, word_count(r))
        ratio = rw / ow
        if ratio < 0.5:
            mask.append(False); reasons.append("too_short"); continue
        if ratio > 2.0:
            mask.append(False); reasons.append("too_long"); continue

        if sim < sim_floor:
            mask.append(False); reasons.append("low_similarity"); continue

        if p < acc_floor:
            mask.append(False); reasons.append("low_acceptability"); continue

        if enforce_single_sentence and sentence_count_heuristic(r) > 1:
            mask.append(False); reasons.append("multi_sentence"); continue

        mask.append(True); reasons.append("valid")
    return mask, reasons


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="matafix/english-sentences-with-sentinel")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--num_samples", type=int, default=10)
    p.add_argument("--seed", type=int, default=43)

    p.add_argument("--base_model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--finetuned_path", type=str, default="grpo_harder_translate_lora/checkpoint-5000")

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--load_in_4bit", action="store_true", default=True)
    p.add_argument("--max_seq_length", type=int, default=1024)

    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.95)

    p.add_argument("--sentinel_model", type=str, default="Prosho/sentinel-src-25")
    p.add_argument("--sentinel_batch_size", type=int, default=16)

    p.add_argument("--embedder_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--cola_model", type=str, default="textattack/roberta-base-CoLA")

    p.add_argument("--sim_floor", type=float, default=0.78)
    p.add_argument("--acc_floor", type=float, default=0.5)
    p.add_argument("--enforce_single_sentence", action="store_true", default=True)

    p.add_argument("--log_file", type=str, default="../evals/TEST.log")
    p.add_argument("--use_intersection", action="store_true", default=True)
    return p.parse_args()

def mean(xs: List[float]) -> float:
    return sum(xs) / max(1, len(xs))

def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    match = re.search(r"checkpoint-(\d+)", args.finetuned_path)
    if match:
        step_str = f"step{match.group(1)}"
    else:
        step_str = "stepFinal"
    base_name, ext = os.path.splitext(args.log_file)
    final_log_file = f"evals/{step_str}_samples{args.num_samples}_{timestamp}{ext}"
    log_dir = os.path.dirname(final_log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    set_all_seeds(args.seed)
    print(f"[Data] Loading {args.dataset}...")
    ds = load_dataset(args.dataset, split=args.split).shuffle(seed=args.seed).select(range(args.num_samples))

    originals = ds["text"] if "text" in ds.column_names else ds["original_text"]
    baseline = ds["sentinel_score"] if "sentinel_score" in ds.column_names else ds["baseline_difficulty"]
    originals = list(originals)
    baseline = list(map(float, baseline))

    sentinel = Sentinel(model_id=args.sentinel_model, device=device, batch_size=args.sentinel_batch_size)
    embedder = Embedder(args.embedder_model, device=device)
    cola = AcceptabilityClassifier(args.cola_model, device=device)

    print(f"[Base] Loading {args.base_model}...")
    base_model, base_tok = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
    )
    
    base_rewrites, base_raw = generate_rewrites(
        base_model, base_tok, originals, device=device,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    del base_model
    torch.cuda.empty_cache()

    print(f"[Fine] Loading {args.finetuned_path}...")
    fine_model, fine_tok = FastLanguageModel.from_pretrained(
        model_name=args.finetuned_path,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
    )
    
    fine_rewrites, fine_raw = generate_rewrites(
        fine_model, fine_tok, originals, device=device,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    del fine_model
    torch.cuda.empty_cache()

    print("[Eval] Similarity + acceptability...")
    base_sims = embedder.cosine_sim(originals, base_rewrites)
    fine_sims = embedder.cosine_sim(originals, fine_rewrites)
    base_acc = cola.p_acceptable(base_rewrites)
    fine_acc = cola.p_acceptable(fine_rewrites)

    base_valid, base_reason = compute_valid_mask(originals, base_rewrites, base_sims, base_acc, args.sim_floor, args.acc_floor, args.enforce_single_sentence)
    fine_valid, fine_reason = compute_valid_mask(originals, fine_rewrites, fine_sims, fine_acc, args.sim_floor, args.acc_floor, args.enforce_single_sentence)

    if args.use_intersection:
        valid = [bv and fv for bv, fv in zip(base_valid, fine_valid)]
        valid_label = "INTERSECTION(base & finetuned)"
    else:
        valid = fine_valid
        valid_label = "FINETUNED valid only"

    idx_valid = [i for i, v in enumerate(valid) if v]
    print(f"[Eval] Valid set {valid_label}: {len(idx_valid)}/{len(originals)}")
    if not idx_valid:
        print("No valid samples. Consider lowering floors or disabling --use_intersection.")
        return

    base_scores = sentinel.difficulty([base_rewrites[i] for i in idx_valid])
    fine_scores = sentinel.difficulty([fine_rewrites[i] for i in idx_valid])
    baseline_valid = [baseline[i] for i in idx_valid]

    base_valid_rate = sum(base_valid) / len(base_valid)
    fine_valid_rate = sum(fine_valid) / len(fine_valid)

    orig_valid = [originals[i] for i in idx_valid]
    base_valid = [base_rewrites[i] for i in idx_valid]
    fine_valid = [fine_rewrites[i] for i in idx_valid]

    # Embed (these are already L2-normalized in embedder)
    E_orig = embedder.encode(orig_valid)
    E_base = embedder.encode(base_valid)
    E_fine = embedder.encode(fine_valid)

    collapse_orig = mean_pairwise_cosine_similarity(E_orig)
    collapse_base = mean_pairwise_cosine_similarity(E_base)
    collapse_fine = mean_pairwise_cosine_similarity(E_fine)

    print(f"[Log] Writing {final_log_file} ...")
    with open(final_log_file, "w", encoding="utf-8") as f:
        f.write("GRPO EVALUATION\n")
        f.write(f"Dataset: {args.dataset} split={args.split} samples={args.num_samples}\n")
        f.write(f"Base model: {args.base_model}\n")
        f.write(f"Finetuned: {args.finetuned_path}\n")
        f.write(f"Constraints: len_ratio[0.5,2.0], sim>={args.sim_floor}, acc>={args.acc_floor}, single_sentence={args.enforce_single_sentence}\n")
        f.write(f"Validity rates: base={base_valid_rate:.3f}, finetuned={fine_valid_rate:.3f}\n")
        f.write(f"Comparison set: {valid_label} count={len(idx_valid)}\n\n")
        f.write("\n" + "="*60)
        f.write(f"\nVALID SET: {valid_label} ({len(idx_valid)}/{len(originals)})\n")
        f.write("="*60)
        f.write(f"\nMean Sentinel (Original baseline): {mean(baseline_valid):.4f}\n")
        f.write(f"Mean Sentinel (Base rewrites):     {mean(base_scores):.4f}\n")
        f.write(f"Mean Sentinel (Finetuned rewrites):{mean(fine_scores):.4f}\n")
        f.write("-"*60)
        f.write(f"\nSentinel Difference (Original - Finetuned): {(mean(baseline_valid)- mean(fine_scores)):.4f}\n")
        f.write(f"Sentinel Difference (Baseline - Finetuned): {(mean(base_scores)- mean(fine_scores)):.4f}\n")
        f.write("-"*60)
        f.write(f"\nValidity rate (Base):             {base_valid_rate:.3f}\n")
        f.write(f"Validity rate (Finetuned):        {fine_valid_rate:.3f}\n")
        f.write(f"Mean pairwise cosine sim (Originals):  {collapse_orig:.4f}\n")
        f.write(f"Mean pairwise cosine sim (Base):       {collapse_base:.4f}\n")
        f.write(f"Mean pairwise cosine sim (Finetuned):  {collapse_fine:.4f}\n")
        f.write("="*60)
        f.write("\nNote: Sentinel higher = easier to translate. Lower rewrite score => harder.\n")

        for j, i in enumerate(idx_valid):
            f.write(f"SAMPLE {j+1} (idx={i})\n{'-'*80}\n")
            f.write(f"[ORIGINAL]  Sentinel={baseline[i]:.4f}\n{originals[i]}\n\n")
            f.write(f"[BASE]  Sentinel={base_scores[j]:.4f}  sim={base_sims[i]:.4f} acc={base_acc[i]:.4f} reason={base_reason[i]}\n")
            f.write(f"{base_rewrites[i]}\n\n")
            f.write(f"[FINETUNED] Sentinel={fine_scores[j]:.4f}  sim={fine_sims[i]:.4f} acc={fine_acc[i]:.4f} reason={fine_reason[i]}\n")
            f.write(f"{fine_rewrites[i]}\n\n")

    print("\n" + "="*60)
    print(f"VALID SET: {valid_label} ({len(idx_valid)}/{len(originals)})")
    print("="*60)
    print(f"Mean Sentinel (Original baseline): {mean(baseline_valid):.4f}")
    print(f"Mean Sentinel (Base rewrites):     {mean(base_scores):.4f}")
    print(f"Mean Sentinel (Finetuned rewrites):{mean(fine_scores):.4f}")
    print("-"*60)
    print(f"Sentinel Difference (Original - Finetuned): {(mean(baseline_valid)- mean(fine_scores)):.4f}")
    print(f"Sentinel Difference (Baseline - Finetuned): {(mean(base_scores)- mean(fine_scores)):.4f}")
    print("-"*60)
    print(f"Validity rate (Base):             {base_valid_rate:.3f}")
    print(f"Validity rate (Finetuned):        {fine_valid_rate:.3f}")
    print(f"Mean pairwise cosine sim (Originals):  {collapse_orig:.4f}")
    print(f"Mean pairwise cosine sim (Base):       {collapse_base:.4f}")
    print(f"Mean pairwise cosine sim (Finetuned):  {collapse_fine:.4f}")
    print("="*60)
    print("Note: Sentinel higher = easier to translate. Lower rewrite score => harder.\n")

if __name__ == "__main__":
    main()
