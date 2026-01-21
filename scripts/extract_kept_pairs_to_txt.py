#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract kept EN-ZH pairs from a JSONL file into a TXT file.

Output format:
en_1
zh_1
en_2
zh_2
...

Only records with kept == True are exported.

Usage:
  python scripts/extract_kept_pairs_to_txt.py \
    --input data/synth_en2zh.jsonl \
    --output data/synth_kept.en_zh.txt

Optional:
  --dedup          # remove exact duplicate (en, zh)
  --shuffle 1337   # shuffle with seed
  --max_pairs 5000 # limit number of exported pairs
"""

import argparse
import json
import random
import sys
from typing import List, Tuple


def _clean_line(s: str) -> str:
    if s is None:
        return ""
    # enforce single line
    return " ".join(str(s).replace("\r", " ").replace("\n", " ").split()).strip()


def read_kept_pairs(path: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                # skip malformed line
                continue

            if obj.get("kept") is not True:
                continue

            en = _clean_line(obj.get("en", ""))
            zh = _clean_line(obj.get("zh", ""))

            if not en or not zh:
                continue

            pairs.append((en, zh))
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input JSONL path.")
    ap.add_argument("--output", required=True, help="Output TXT path.")
    ap.add_argument("--dedup", action="store_true", help="Deduplicate exact (en, zh) pairs.")
    ap.add_argument("--shuffle", type=int, default=None, help="Shuffle with a given seed (e.g., 1337).")
    ap.add_argument("--max_pairs", type=int, default=None, help="Max number of pairs to export.")
    args = ap.parse_args()

    pairs = read_kept_pairs(args.input)

    if args.dedup:
        seen = set()
        uniq: List[Tuple[str, str]] = []
        for en, zh in pairs:
            key = (en, zh)
            if key in seen:
                continue
            seen.add(key)
            uniq.append(key)
        pairs = uniq

    if args.shuffle is not None:
        rnd = random.Random(args.shuffle)
        rnd.shuffle(pairs)

    if args.max_pairs is not None:
        pairs = pairs[: args.max_pairs]

    with open(args.output, "w", encoding="utf-8") as out:
        for en, zh in pairs:
            out.write(en + "\n")
            out.write(zh + "\n")

    print(f"Done. exported_pairs={len(pairs)} -> {args.output}")


if __name__ == "__main__":
    main()
