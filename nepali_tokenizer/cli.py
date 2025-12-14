import argparse
import sys
from typing import Optional

import json
from .tokenizer import NepaliTokenizer, RuleConfig


def build_config(args: argparse.Namespace) -> RuleConfig:
    return RuleConfig(
        enable_sentence_segmentation=not args.no_sentence,
        enable_word_segmentation=True,
        enable_postpositions=not args.no_case,
        enable_genitives=not args.no_genitive,
        enable_plural=not args.no_plural,
        enable_particles=not args.no_particles,
        enable_verb_morphology=not args.no_verbs,
        enable_pronoun_inflection=not args.no_pronouns,
        enable_sandhi=not args.no_sandhi,
        enable_compounds=not args.no_compounds,
    )


def read_input(file: Optional[str]) -> str:
    if file:
        with open(file, "r", encoding="utf-8") as f:
            return f.read()
    return sys.stdin.read()


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="nepali-tokenize",
        description="Deterministic grammar-based Nepali tokenizer",
    )
    parser.add_argument("file", nargs="?", help="Input file (UTF-8). If omitted, reads from stdin.")
    parser.add_argument("--flat", action="store_true", help="Output flat tokens only (default).")
    parser.add_argument("--hier", action="store_true", help="Output hierarchical analyses per token.")
    parser.add_argument("--format", choices=["plain", "json", "jsonl", "conll"], help="Output format.")
    parser.add_argument("--compounds", help="Path to JSON file for external compound decompositions.")
    parser.add_argument("--sandhi", help="Path to JSON file for extra sandhi patterns.")
    parser.add_argument("--verbs", help="Path to JSON file for extra verb morphemes.")
    parser.add_argument("--profile", action="store_true", help="Print simple timing profile per stage.")
    parser.add_argument("--debug", action="store_true", help="Emit per-token rule decision trace to stderr.")
    # Toggles to disable specific rules
    parser.add_argument("--no-sentence", action="store_true", help="Disable sentence segmentation.")
    parser.add_argument("--no-case", action="store_true", help="Disable case/postposition suffix splits.")
    parser.add_argument("--no-genitive", action="store_true", help="Disable genitive splits.")
    parser.add_argument("--no-plural", action="store_true", help="Disable plural suffix splits.")
    parser.add_argument("--no-particles", action="store_true", help="Disable particle splits.")
    parser.add_argument("--no-verbs", action="store_true", help="Disable verb morphology splits.")
    parser.add_argument("--no-pronouns", action="store_true", help="Disable pronoun POS hints.")
    parser.add_argument("--no-sandhi", action="store_true", help="Disable sandhi handling.")
    parser.add_argument("--no-compounds", action="store_true", help="Disable compound handling.")

    args = parser.parse_args(argv)
    text = read_input(args.file)
    config = build_config(args)
    extra_compounds = None
    extra_sandhi = None
    extra_verbs = None
    if args.compounds:
        with open(args.compounds, "r", encoding="utf-8") as f:
            extra_compounds = json.load(f)
    if args.sandhi:
        with open(args.sandhi, "r", encoding="utf-8") as f:
            extra_sandhi = json.load(f)
    if args.verbs:
        with open(args.verbs, "r", encoding="utf-8") as f:
            extra_verbs = json.load(f)
    tokenizer = NepaliTokenizer(
        config,
        extra_compounds=extra_compounds,
        extra_sandhi=extra_sandhi,
        extra_verbs=extra_verbs,
        debug=args.debug,
    )

    import time
    t0 = time.perf_counter()
    tokens, analyses = tokenizer.tokenize(text, hierarchical=args.hier)
    t1 = time.perf_counter()

    fmt = args.format or ("plain" if not args.hier else "conll")
    if fmt == "plain":
        for tok in tokens:
            print(tok)
    elif fmt == "conll":
        # token \t root \t suffixes \t pos
        if analyses is None:
            for tok in tokens:
                print(f"{tok}\t\t\t")
        else:
            for tok, ana in zip(tokens, analyses):
                root = ana.root or ""
                suff = ",".join(ana.suffixes) if ana.suffixes else ""
                pos = ana.pos or ""
                print(f"{tok}\t{root}\t{suff}\t{pos}")
    elif fmt == "json":
        import json as _json
        out = []
        if analyses is None:
            for tok in tokens:
                out.append({"token": tok})
        else:
            for tok, ana in zip(tokens, analyses):
                out.append({
                    "token": tok,
                    "root": ana.root,
                    "suffixes": ana.suffixes,
                    "pos": ana.pos,
                })
        print(_json.dumps({"tokens": out}, ensure_ascii=False))
    elif fmt == "jsonl":
        import json as _json
        if analyses is None:
            for tok in tokens:
                print(_json.dumps({"token": tok}, ensure_ascii=False))
        else:
            for tok, ana in zip(tokens, analyses):
                print(_json.dumps({
                    "token": tok,
                    "root": ana.root,
                    "suffixes": ana.suffixes,
                    "pos": ana.pos,
                }, ensure_ascii=False))

    if args.profile:
        total = t1 - t0
        print(f"# profile: tokenize={total:.6f}s tokens={len(tokens)}", file=sys.stderr)
    if args.debug:
        for line in tokenizer.get_trace():
            print(f"# debug: {line}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
