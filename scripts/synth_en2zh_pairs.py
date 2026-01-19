#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Synthetic EN-ZH parallel data generation (from scratch) using an open-source LLM on a single GPU.

Output: JSONL (one record per line), fields include:
  id, en, zh, domain, difficulty, tags, kept, gen_params, judge(optional), flags

Recommended deps:
  pip install -U "transformers>=4.41.0" accelerate bitsandbytes sentencepiece

Example:
  python scripts/synth_en2zh_pairs.py \
    --output data/synth_en2zh.jsonl \
    --n_samples 5000 \
    --model Qwen/Qwen2.5-3B-Instruct \
    --quant4 \
    --batch_size 16 \
    --temperature 0.8 --top_p 0.95 \
    --max_new_tokens 160 \
    --judge \
    --seed 1337

For a resumed generation, eg:
    python scripts/synth_en2zh_pairs.py \
    --output data/synth_en2zh.jsonl \
    --n_samples 5000 \
    --model Qwen/Qwen2.5-3B-Instruct \
    --quant4 \
    --batch_size 16 \
    --temperature 0.8 --top_p 0.95 \
    --max_new_tokens 160 \
    --judge \
    --resume \
    --seed 1338
    
Notes:
- From-scratch generation has higher noise risk than "given EN -> translate to ZH".
  The script includes rule-based filters and optional LLM judge to mitigate noise.
"""

import argparse
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from transformers import BitsAndBytesConfig
    _HAS_BNB = True
except Exception:
    _HAS_BNB = False

CJK_RE = re.compile(r"[\u4e00-\u9fff]")  # basic CJK Unified Ideographs
DIGIT_RE = re.compile(r"\d+")
JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)

ABBREV_DOT_RE = re.compile(r"\b(?:Dr|Mr|Ms|Mrs|Prof|Sr|Jr|St)\.", re.IGNORECASE)
ACRONYM_DOTS_RE = re.compile(r"\b(?:[A-Za-z]\.){2,}")  # e.g., U.S., U.K., Ph.D.


# ----------------------------
# Config / Buckets
# ----------------------------
DEFAULT_DOMAINS = [
    ("ted_talk", 0.65),
    ("interview", 0.20),
    ("casual_dialogue", 0.10),
    ("lecture", 0.03),
    ("news_like", 0.02),
]

DEFAULT_DIFFICULTY = [
    ("easy", 0.30),
    ("medium", 0.50),
    ("hard", 0.20),
]

DEFAULT_TAGS = [
    ("numbers_units", 0.20),
    ("named_entities", 0.20),
    ("long_sentence", 0.15),
    ("coreference", 0.15),
    ("spoken_style", 0.15),
    ("idiom", 0.10),
    ("none", 0.05),
]


@dataclass
class GenParams:
    temperature: float
    top_p: float
    top_k: int
    max_new_tokens: int
    do_sample: bool
    num_beams: int
    repetition_penalty: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def weighted_choice(items: List[Tuple[str, float]]) -> str:
    r = random.random()
    acc = 0.0
    for name, w in items:
        acc += w
        if r <= acc:
            return name
    return items[-1][0]


def extract_digits(s: str) -> List[str]:
    return DIGIT_RE.findall(s or "")


def contains_cjk(s: str) -> bool:
    return bool(CJK_RE.search(s or ""))


def basic_clean(s: str) -> str:
    s = (s or "").strip()
    # strip code fences
    if s.startswith("```"):
        s = s.strip("`").strip()
    return s


def safe_json_extract(text: str) -> Optional[Dict[str, Any]]:
    """
    Try best-effort JSON object extraction from model output.
    """
    if not text:
        return None
    text = text.strip()
    # If model already output pure JSON, parse directly
    try:
        return json.loads(text)
    except Exception:
        pass

    # Else try to find the first {...} block
    m = JSON_OBJ_RE.search(text)
    if not m:
        return None
    blob = m.group(0).strip()
    # Remove trailing junk after last brace if any (rare)
    # Try parse
    try:
        return json.loads(blob)
    except Exception:
        # Sometimes single quotes appear; attempt minimal normalization
        blob2 = blob.replace("\n", " ")
        blob2 = re.sub(r"(\w+)\s*:", r'"\1":', blob2)  # risky but helps when keys are unquoted
        blob2 = blob2.replace("'", '"')
        try:
            return json.loads(blob2)
        except Exception:
            return None


def build_gen_prompt(domain: str, difficulty: str, tag: str) -> List[Dict[str, str]]:
    """
    Return chat messages for generation (one EN sentence + its ZH translation).
    """
    system = (
        "You are a high-quality dataset generator for machine translation.\n"
        "Task: Create ONE English sentence and its faithful Simplified Chinese translation.\n"
        "Style: conversational spoken English (TED/interview-like), natural and story-friendly; avoid textbook or encyclopedic tone.\n"
        "Output MUST be STRICT JSON (no markdown, no explanations, no extra text).\n"
        "JSON schema:\n"
        "{\n"
        '  "en": string,        // English sentence\n'
        '  "zh": string,        // Simplified Chinese translation of "en"\n'
        '  "domain": string,    // one of: ted_talk, interview, lecture, casual_dialogue, news_like\n'
        '  "difficulty": string,// one of: easy, medium, hard\n'
        '  "tags": [string]     // list of tags describing the sentence\n'
        "}\n"
        "Constraints:\n"
        "1) The Chinese MUST translate the English exactly: no added/omitted facts.\n"
        "2) Preserve numbers, dates, units, and named entities consistently.\n"
        "3) Produce natural, fluent text in both languages.\n"
        "4) Output 1-3 short sentences at most. Do not output long multi-sentence paragraphs.\n"
        "5) Do not use quotation marks to wrap the entire sentence.\n"
        "6) Avoid excessive sentence splits; prefer 1-3 sentences total. Do not use bullet points.\n"
        "7) Do NOT include any newline characters in either \"en\" or \"zh\".\n"
    )

    # Bucket-specific requirements
    req = []
    req.append(f'Domain style: "{domain}".')
    req.append(f'Difficulty: "{difficulty}".')
    if tag == "numbers_units":
        req.append("MUST include at least ONE Arabic numeral (0-9) in the English text and preserve the SAME VALUE in Chinese "
                    "also include at least one unit/measure or identifier (e.g., %, km, miles, years, °C, dollars, minutes, Exit 238, I-40).")
    elif tag == "named_entities":
        req.append("Include at least ONE named entity (person/place/org) naturally in context, "
                   "and keep the entity consistent between English and Chinese (do not change the referenced person/place/organization).")
    elif tag == "long_sentence":
        req.append("Make it longer: at least 25 words in English; keep it within 1-3 sentences total.")
    elif tag == "coreference":
        req.append("Include a pronoun/coreference that remains clear in Chinese.")
    elif tag == "spoken_style":
        req.append("Use TED/interview spoken style (contractions, discourse markers) and keep it within 1-3 sentences.")
    elif tag == "idiom":
        req.append("Include one idiomatic expression and translate it naturally (not word-by-word).")
    else:
        req.append("No special constraints beyond faithfulness and fluency.")

    user = (
        "Generate ONE sample now.\n"
        + " ".join(req)
        + "\nRemember: Output ONLY valid JSON."
    )

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_judge_prompt(en: str, zh: str) -> List[Dict[str, str]]:
    """
    LLM judge: check semantic fidelity, omissions/additions, digit/entity consistency.
    Output strict JSON.
    """
    system = (
        "You are a strict bilingual translation quality judge.\n"
        "Evaluate whether the Simplified Chinese translation is semantically equivalent to the English.\n"
        "Output MUST be STRICT JSON only, no extra text.\n"
        "JSON schema:\n"
        "{\n"
        '  "pass": boolean,\n'
        '  "score": integer,        // 0-100\n'
        '  "issues": [string]       // short issue labels\n'
        "}\n"
        "Scoring rubric (calibrated):\n"
        "- 90-100: perfect or near-perfect equivalence.\n"
        "- 80-89: equivalent with minor stylistic differences.\n"
        "- 60-79: noticeable issues but core meaning mostly preserved.\n"
        "- 0-59: mistranslation, missing/added key info, wrong numbers/units/entities.\n"
        "Fail (pass=false) if any key information is added/omitted, or numbers/units/entities are wrong.\n"
        "Do NOT fail for acceptable numeric paraphrases (e.g., ‘50 billion’ vs ‘50,000,000,000’ vs ‘500亿’), as long as the value is consistent.\n"
    )

    user = f'English: {en}\nChinese: {zh}\nReturn JSON now.'
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def make_prompt(tokenizer: AutoTokenizer, messages: List[Dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    # Fallback manual template
    parts = []
    for m in messages:
        parts.append(f"{m['role'].upper()}:\n{m['content']}\n")
    parts.append("ASSISTANT:\n")
    return "\n".join(parts)


def load_model_and_tokenizer(model_name: str, quant4: bool) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: Dict[str, Any] = dict(device_map="auto", torch_dtype=torch.float16)

    if quant4:
        if not _HAS_BNB:
            raise RuntimeError("bitsandbytes not available. Install: pip install bitsandbytes")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        kwargs["quantization_config"] = bnb_config

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    return tokenizer, model


@torch.inference_mode()
def generate_text_batch(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    prompts: List[str],
    gen_params: GenParams,
) -> List[str]:
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    )
    input_ids = enc["input_ids"].to(model.device)
    attn_mask = enc["attention_mask"].to(model.device)

    gen_kwargs = dict(
        max_new_tokens=gen_params.max_new_tokens,
        do_sample=gen_params.do_sample,
        temperature=gen_params.temperature if gen_params.do_sample else None,
        top_p=gen_params.top_p if gen_params.do_sample else None,
        top_k=gen_params.top_k if gen_params.do_sample else None,
        num_beams=gen_params.num_beams,
        repetition_penalty=gen_params.repetition_penalty,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

    out = model.generate(
        input_ids=input_ids,
        attention_mask=attn_mask,
        **gen_kwargs
    )

    results: List[str] = []
    prompt_lens = attn_mask.sum(dim=1).tolist()

    for i in range(out.size(0)):
        prompt_len = int(prompt_lens[i])
        gen_ids = out[i, prompt_len:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        results.append(text.strip())
    return results


def rule_filter(
    en: str,
    zh: str,
    tag: str,
    min_en_chars: int,
    min_zh_chars: int,
    max_len_ratio: float,
    strict_digits: bool,
) -> List[str]:
    flags: List[str] = []

    en = (en or "").strip()
    zh = (zh or "").strip()

    if len(en) < min_en_chars:
        flags.append("en_too_short")
    if len(zh) < min_zh_chars:
        flags.append("zh_too_short")

    if not contains_cjk(zh):
        flags.append("no_cjk")

    # Length ratio heuristic
    if en and zh:
        ratio = (len(zh) + 1e-6) / (len(en) + 1e-6)
        if ratio > max_len_ratio:
            flags.append("len_ratio_too_high")

    # Digit constraints
    en_digits = extract_digits(en)
    zh_digits = extract_digits(zh)

    if tag == "numbers_units" and not en_digits:
        flags += ["tag_requires_digits_but_en_has_none", "DROP"]


    if en_digits:
        missing = [d for d in en_digits if d not in zh_digits]
        if missing:
            flags.append("digits_mismatch")
            if strict_digits:
                flags.append("DROP")


    # Remove abbreviation dots before counting sentence endings
    _en = en
    _en = ABBREV_DOT_RE.sub(lambda m: m.group(0).replace(".", ""), _en)   # Dr. -> Dr
    _en = ACRONYM_DOTS_RE.sub(lambda m: m.group(0).replace(".", ""), _en) # U.S. -> US

    # Normalize ellipsis so it won't be counted as 3 sentence endings
    _en = _en.replace("...", "…")
    _zh = zh.replace("……", "…")  # common Chinese ellipsis

    # Remove ellipsis for end-mark counting
    _en_for_count = _en.replace("…", "")
    _zh_for_count = _zh.replace("…", "")

    # Count sentence-ending punctuation, excluding dots that are part of an ellipsis
    en_end = len(re.findall(r"(?<!\.)[.!?](?!\.)", _en_for_count))
    zh_end = len(re.findall(r"[。！？]", _zh_for_count))

    # You currently want 1-3 short sentences at most; drop only if >3
    if en_end > 3:
        flags += ["en_multi_sentence", "DROP"]
    if zh_end > 3:
        flags += ["zh_multi_sentence", "DROP"]

    # Also drop if newline sneaks in
    if "\n" in en or "\r" in en:
        flags += ["en_has_newline", "DROP"]
    if "\n" in zh or "\r" in zh:
        flags += ["zh_has_newline", "DROP"]

    return flags


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, help="Output JSONL path.")
    ap.add_argument("--n_samples", type=int, default=1000, help="Number of target samples to GENERATE (not guaranteed kept).")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct", help="HF model name or local path.")
    ap.add_argument("--quant4", action="store_true", help="Enable 4-bit quantization (recommended for RTX 4060).")
    ap.add_argument("--batch_size", type=int, default=8, help="Batch size (reduce if OOM).")
    ap.add_argument("--max_new_tokens", type=int, default=220, help="Max new tokens for JSON generation.")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--num_beams", type=int, default=1, help=">1 for beam search (do_sample disabled).")
    ap.add_argument("--repetition_penalty", type=float, default=1.05)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--resume", action="store_true", help="Resume generation (skip already written lines).")

    # Filtering
    ap.add_argument("--min_en_chars", type=int, default=12)
    ap.add_argument("--min_zh_chars", type=int, default=6)
    ap.add_argument("--max_len_ratio", type=float, default=3.8)
    ap.add_argument("--strict_digits", action="store_true", help="Drop if digits mismatch.")

    # Optional LLM judge
    ap.add_argument("--judge", action="store_true", help="Enable LLM judge for fidelity checking (slower, higher quality).")
    ap.add_argument("--judge_pass_threshold", type=int, default=75, help="If judge pass=True but score<threshold -> drop.")

    args = ap.parse_args()
    set_seed(args.seed)

    if args.quant4 and not _HAS_BNB:
        print("ERROR: bitsandbytes not installed but --quant4 enabled. Run: pip install bitsandbytes", file=sys.stderr)
        sys.exit(1)

    tokenizer, model = load_model_and_tokenizer(args.model, args.quant4)

    do_sample = (args.num_beams == 1)
    gen_params = GenParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        do_sample=do_sample,
        num_beams=args.num_beams,
        repetition_penalty=args.repetition_penalty,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    start_idx = 0
    if args.resume and os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            start_idx = sum(1 for _ in f)
        print(f"[RESUME] Found {start_idx} existing lines; continue from id={start_idx}")

    target_total = args.n_samples
    t0 = time.time()

    kept_count = 0
    gen_count = 0

    # Simple stats
    bucket_stats: Dict[str, int] = {}
    drop_reasons: Dict[str, int] = {}

    def bump(d: Dict[str, int], k: str, v: int = 1) -> None:
        d[k] = d.get(k, 0) + v

    with open(args.output, "a", encoding="utf-8") as fout:
        next_id = start_idx

        total_lines_target = start_idx + target_total
        pbar = tqdm(total=total_lines_target, initial=start_idx, desc="Generating", unit="samples")
        last_t = time.time()
        last_gen = gen_count
        last_kept = kept_count

        while gen_count < target_total:
            batch_start_gen = gen_count
            batch_start_kept = kept_count
            # Prepare one batch of buckets
            batch_meta: List[Tuple[int, str, str, str]] = []
            for _ in range(args.batch_size):
                if gen_count >= target_total:
                    break
                domain = weighted_choice(DEFAULT_DOMAINS)
                difficulty = weighted_choice(DEFAULT_DIFFICULTY)
                tag = weighted_choice(DEFAULT_TAGS)
                batch_meta.append((next_id, domain, difficulty, tag))
                next_id += 1
                gen_count += 1

            # Build prompts
            gen_prompts = [
                make_prompt(tokenizer, build_gen_prompt(domain, difficulty, tag))
                for (_, domain, difficulty, tag) in batch_meta
            ]

            # Generate JSON candidates
            raw_out = generate_text_batch(tokenizer, model, gen_prompts, gen_params)

            # ---------------------------------------------------------
            # Pre-parse + rule_filter FIRST, then judge ONLY non-DROP
            # ---------------------------------------------------------
            judge_results: List[Optional[Dict[str, Any]]] = [None] * len(batch_meta)

            # Cache parsed content so we don't run rule_filter twice
            parsed_obj: List[Optional[Dict[str, Any]]] = [None] * len(batch_meta)
            parsed_en: List[str] = [""] * len(batch_meta)
            parsed_zh: List[str] = [""] * len(batch_meta)
            pre_flags_list: List[List[str]] = [[] for _ in range(len(batch_meta))]

            for i, ((rid, domain, difficulty, tag), out_text) in enumerate(zip(batch_meta, raw_out)):
                obj = safe_json_extract(out_text)
                parsed_obj[i] = obj

                if not obj or "en" not in obj or "zh" not in obj:
                    # treat as drop; writing loop will mark json_parse_fail
                    pre_flags_list[i] = ["json_parse_fail", "DROP"]
                    continue

                en = basic_clean(str(obj.get("en", "")))
                zh = basic_clean(str(obj.get("zh", "")))
                parsed_en[i] = en
                parsed_zh[i] = zh

                pre_flags = rule_filter(
                    en=en, zh=zh, tag=tag,
                    min_en_chars=args.min_en_chars,
                    min_zh_chars=args.min_zh_chars,
                    max_len_ratio=args.max_len_ratio,
                    strict_digits=args.strict_digits,
                )
                pre_flags_list[i] = pre_flags

            # Optionally judge only for those WITHOUT DROP
            if args.judge:
                judge_prompts: List[str] = []
                judge_map: List[int] = []  # map judge output back to sample index i

                for i, ((rid, domain, difficulty, tag), out_text) in enumerate(zip(batch_meta, raw_out)):
                    if "DROP" in pre_flags_list[i]:
                        continue  # skip judge for obviously bad samples
                    en = parsed_en[i]
                    zh = parsed_zh[i]
                    # extra guard
                    if not en or not zh:
                        continue

                    judge_prompts.append(make_prompt(tokenizer, build_judge_prompt(en, zh)))
                    judge_map.append(i)

                if judge_prompts:
                    judge_raw = generate_text_batch(
                        tokenizer,
                        model,
                        judge_prompts,
                        GenParams(
                            temperature=0.0, top_p=1.0, top_k=0,
                            max_new_tokens=180,
                            do_sample=False,
                            num_beams=1,
                            repetition_penalty=1.0
                        )
                    )
                    for k, jr in enumerate(judge_raw):
                        i = judge_map[k]
                        judge_results[i] = safe_json_extract(jr)

            # Write records
            for i, ((rid, domain, difficulty, tag), out_text) in enumerate(zip(batch_meta, raw_out)):
                obj = parsed_obj[i]
                flags = list(pre_flags_list[i])
                record: Dict[str, Any] = {
                    "id": rid,
                    "domain": domain,
                    "difficulty": difficulty,
                    "tags": [tag],
                    "model": args.model,
                    "gen_params": asdict(gen_params),
                    "raw": out_text,
                }

                if not obj or "en" not in obj or "zh" not in obj:
                    # keep your existing json_parse_fail writing logic
                    record.update({"en": "", "zh": "", "kept": False, "flags": ["json_parse_fail"]})
                    bump(drop_reasons, "json_parse_fail")
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    continue

                en = parsed_en[i]
                zh = parsed_zh[i]

                # If model outputs different bucket labels, keep ours as "target_bucket" and store "model_bucket"
                record["model_bucket"] = {
                    "domain": obj.get("domain", None),
                    "difficulty": obj.get("difficulty", None),
                    "tags": obj.get("tags", None),
                }

                kept = True
                if "DROP" in flags:
                    kept = False
                    flags = [f for f in flags if f != "DROP"]

                # Judge check
                if args.judge and kept:
                    jr = judge_results[i]
                    record["judge"] = jr
                    if not jr or "pass" not in jr or "score" not in jr:
                        kept = False
                        flags.append("judge_parse_fail")
                        bump(drop_reasons, "judge_parse_fail")
                    else:
                        if (jr.get("pass") is not True) or (int(jr.get("score", 0)) < args.judge_pass_threshold):
                            kept = False
                            flags.append("judge_reject")

                # Final
                record.update({"en": en, "zh": zh, "kept": kept, "flags": flags})

                bump(bucket_stats, f"{domain}|{difficulty}|{tag}")
                if not kept:
                    # pick one primary reason
                    if "judge_reject" in flags:
                        bump(drop_reasons, "judge_reject")
                    elif "tag_requires_digits_but_en_has_none" in flags:
                        bump(drop_reasons, "tag_requires_digits_but_en_has_none")
                    elif "json_parse_fail" in flags:
                        bump(drop_reasons, "json_parse_fail")
                    elif "en_multi_sentence" in flags:
                        bump(drop_reasons, "en_multi_sentence")
                    elif "zh_multi_sentence" in flags:
                        bump(drop_reasons, "zh_multi_sentence")
                    else:
                        bump(drop_reasons, flags[0])

                if kept:
                    kept_count += 1

                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

            fout.flush()
            torch.cuda.empty_cache()

            # update progress print
            batch_gen = gen_count - batch_start_gen
            batch_kept = kept_count - batch_start_kept
            pbar.update(batch_gen)

            now = time.time()
            dt = max(now - last_t, 1e-6)
            gen_delta = gen_count - last_gen
            kept_delta = kept_count - last_kept
            speed = gen_delta / dt
            keep_rate = (kept_count / gen_count) if gen_count else 0.0

            pbar.set_postfix({
                "kept": kept_count,
                "keep_rate": f"{keep_rate:.2f}",
                "speed/s": f"{speed:.2f}",
                "batch_kept": batch_kept,
            })

            last_t, last_gen, last_kept = now, gen_count, kept_count
        pbar.close()

    elapsed = time.time() - t0
    print(f"\nDone. generated={gen_count}, kept={kept_count}, elapsed_sec={elapsed:.1f}")
    if bucket_stats:
        print("\nTop buckets:")
        top = sorted(bucket_stats.items(), key=lambda x: x[1], reverse=True)[:10]
        for k, v in top:
            print(f"  {k}: {v}")
    if drop_reasons:
        print("\nDrop reasons (top):")
        topd = sorted(drop_reasons.items(), key=lambda x: x[1], reverse=True)[:15]
        for k, v in topd:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
