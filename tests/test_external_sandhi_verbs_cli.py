import json
import subprocess
import sys
import tempfile
from pathlib import Path


def run_cli(args, input_text):
    # Run the installed CLI via -m to ensure local package usage
    cmd = [sys.executable, "-m", "nepali_tokenizer.cli"] + args
    proc = subprocess.run(cmd, input=input_text.encode("utf-8"), capture_output=True)
    return proc.returncode, proc.stdout.decode("utf-8"), proc.stderr.decode("utf-8")


def test_cli_loads_external_sandhi_and_verbs(tmp_path: Path):
    # Prepare external JSON files
    sandhi_list = ["उनीले", "योमा"]
    verbs_list = ["गइरहेको", "हुँदै"]
    sandhi_file = tmp_path / "sandhi.json"
    verbs_file = tmp_path / "verbs.json"
    sandhi_file.write_text(json.dumps(sandhi_list, ensure_ascii=False), encoding="utf-8")
    verbs_file.write_text(json.dumps(verbs_list, ensure_ascii=False), encoding="utf-8")

    # Input text includes a sandhi form and a verb morpheme form
    text = "उनीले गइरहेको छ।"

    # Ask for hierarchical output to inspect analyses
    rc, out, err = run_cli([
        "--hier",
        "--format", "conll",
        "--sandhi", str(sandhi_file),
        "--verbs", str(verbs_file),
    ], text)

    assert rc == 0, err
    lines = [l for l in out.splitlines() if l.strip()]
    # Expect sandhi to split into parts and verbs recognized
    tokens = [l.split("\t")[0] for l in lines]
    assert "उनी" in tokens
    assert "ले" in tokens
    assert "गइरहेको" in tokens or "गइरह" in tokens or "गइ" in tokens

    # Ensure debug trace is emitted when --debug is set (optional behavior)
    rc, out2, err2 = run_cli([
        "--hier",
        "--format", "conll",
        "--sandhi", str(sandhi_file),
        "--verbs", str(verbs_file),
        "--debug",
    ], text)
    assert rc == 0
    assert any("# debug:" in line for line in err2.splitlines())
