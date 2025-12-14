import argparse
import json
from pathlib import Path

from nepali_tokenizer.tokenizer import NepaliTokenizer, RuleConfig


def load_gold(path: str):
    # Expect JSONL lines: {"text": ..., "tokens": ["..."]}
    gold = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        gold.append(obj)
    return gold


def to_boundaries(text: str, tokens: list[str]) -> set[tuple[int, int]]:
    bounds = set()
    i = 0
    for t in tokens:
        # Find t at position i; since gold/pred are derived from the same text, we assume sequential match
        j = i + len(t)
        bounds.add((i, j))
        i = j
    return bounds


def evaluate(gold, hier: bool = False):
    tok = NepaliTokenizer(RuleConfig())
    tp = fp = fn = 0
    total_docs = 0
    # Category-wise counters if gold provides types
    cat_counts = {}
    for item in gold:
        text = item["text"]
        gold_tokens = item["tokens"]
        gold_types = item.get("types")
        pred, _ = tok.tokenize(text, hierarchical=hier)
        gold_b = to_boundaries(text, gold_tokens)
        pred_b = to_boundaries(text, pred)
        tp += len(gold_b & pred_b)
        fp += len(pred_b - gold_b)
        fn += len(gold_b - pred_b)
        total_docs += 1
        # Category metrics: align lengths, then compare types
        if gold_types and len(gold_types) == len(gold_tokens):
            # Get predicted types via a simple heuristic: classify suffix when applicable else token type
            pred_types = []
            for t in pred:
                # Quick classify using tokenizer.classify_suffix for suffix categories else default 'word'
                # Note: this is naive without full per-token analysis; good enough for coarse metrics
                c = tok.classify_suffix(t)
                if c == "suffix":
                    # Fallback categories for punctuation, numerals, english
                    if t in {"।", "?", "!", "॥", ",", ";", ":", "(", ")", "[", "]", "{", "}", '"', "'", "—", "-", "…"}:
                        c = "punctuation"
                    else:
                        c = "word"
                pred_types.append(c)
            for g_t, p_t in zip(gold_types, pred_types):
                cat_counts.setdefault(g_t, {"correct": 0, "total": 0})
                cat_counts[g_t]["total"] += 1
                if g_t == p_t:
                    cat_counts[g_t]["correct"] += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    result = {
        "docs": total_docs,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    if cat_counts:
        result["categories"] = {
            k: {
                "accuracy": (v["correct"] / v["total"] if v["total"] else 0.0),
                "total": v["total"],
            }
            for k, v in cat_counts.items()
        }
    print(json.dumps(result, ensure_ascii=False))


def evaluate_whitespace(gold):
    # Baseline: whitespace split
    tp = fp = fn = 0
    for item in gold:
        text = item["text"]
        gold_tokens = item["tokens"]
        pred = text.split()
        gold_b = to_boundaries(text, gold_tokens)
        pred_b = to_boundaries(text, pred)
        tp += len(gold_b & pred_b)
        fp += len(pred_b - gold_b)
        fn += len(gold_b - pred_b)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def main():
    ap = argparse.ArgumentParser(description="Evaluate tokenizer against a gold tokenization JSONL")
    ap.add_argument("gold", help="Path to JSONL with fields: text, tokens[]")
    ap.add_argument("--hier", action="store_true", help="Include hierarchical analysis during prediction")
    ap.add_argument("--baseline", choices=["whitespace"], help="Compare against a baseline tokenizer")
    args = ap.parse_args()
    gold = load_gold(args.gold)
    if args.baseline == "whitespace":
        b = evaluate_whitespace(gold)
        print(json.dumps({"baseline": "whitespace", **b}, ensure_ascii=False))
    evaluate(gold, hier=args.hier)


if __name__ == "__main__":
    main()
