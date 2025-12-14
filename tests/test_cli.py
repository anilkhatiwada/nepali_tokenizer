import io
import sys

from nepali_tokenizer.cli import main


def run_cli(input_text: str, args=None) -> str:
    buf_in = io.StringIO(input_text)
    buf_out = io.StringIO()
    old_stdin, old_stdout = sys.stdin, sys.stdout
    try:
        sys.stdin = buf_in
        sys.stdout = buf_out
        code = main(args or [])
        assert code == 0
        return buf_out.getvalue().strip()
    finally:
        sys.stdin, sys.stdout = old_stdin, old_stdout


def test_cli_hier_compounds_and_verbs():
    out = run_cli("विद्यालयमा पढ्दै छ।", ["--hier"])
    # Expect multiple lines; first two lines should be compound parts
    lines = out.splitlines()
    assert lines[0].startswith("विद्या")
    assert lines[1].startswith("लय")
    # Ensure verb root appears on some line
    assert any("\tपढ\t" in line for line in lines)


def test_cli_flat_disable_rules():
    out = run_cli(
        "विद्यालयमा पढ्दै छ।",
        ["--no-compounds", "--no-verbs", "--no-case", "--no-genitive", "--no-plural", "--no-particles"],
    )  # flat default
    # With compounds/verbs/case-related splits disabled, whole words should remain
    toks = out.splitlines()
    assert "विद्यालयमा" in toks
    assert "पढ्दै" in toks
    assert "छ" in toks
