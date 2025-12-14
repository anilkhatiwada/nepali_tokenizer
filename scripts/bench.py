import argparse
import time
from pathlib import Path

from nepali_tokenizer.tokenizer import NepaliTokenizer, RuleConfig


def run_bench(text: str, iters: int, hier: bool) -> None:
    tok = NepaliTokenizer(RuleConfig())
    # Warmup
    tok.tokenize(text, hierarchical=hier)
    start = time.perf_counter()
    total_tokens = 0
    for _ in range(iters):
        flat, _ = tok.tokenize(text, hierarchical=hier)
        total_tokens += len(flat)
    elapsed = time.perf_counter() - start
    toks_per_sec = total_tokens / elapsed if elapsed > 0 else float('inf')
    print(f"iters={iters} tokens={total_tokens} time={elapsed:.3f}s rate={toks_per_sec:.0f} toks/s")


def main():
    ap = argparse.ArgumentParser(description="Benchmark Nepali tokenizer throughput")
    ap.add_argument("file", help="Input text file (UTF-8)")
    ap.add_argument("--iters", type=int, default=50, help="Iterations for timing loop")
    ap.add_argument("--hier", action="store_true", help="Include hierarchical analysis in tokenization")
    args = ap.parse_args()
    text = Path(args.file).read_text(encoding="utf-8")
    run_bench(text, args.iters, args.hier)


if __name__ == "__main__":
    main()
