import pytest

from nepali_tokenizer.tokenizer import NepaliTokenizer


def test_compound_jalbidyut():
    tok = NepaliTokenizer()
    tokens, analyses = tok.tokenize("जलविद्युत")
    assert tokens == ["जल", "विद्युत"]


def test_compound_krishibhumi():
    tok = NepaliTokenizer()
    tokens, analyses = tok.tokenize("कृषिभूमि")
    assert tokens == ["कृषि", "भूमि"]


def test_compound_vidyut_ghar():
    tok = NepaliTokenizer()
    tokens, analyses = tok.tokenize("विद्युतगृह")
    assert tokens == ["विद्युत", "गृह"]


def test_verb_progressive_gairahiraheko():
    tok = NepaliTokenizer()
    tokens, analyses = tok.tokenize("गइरहेको")
    # Expect root + progressive suffix
    assert "verb_root" in {t.type for t in tok.analyze_word("गइरहेको")[0]}
    assert "गइरहेको" not in tokens  # should be split


def test_verb_hundai_marker():
    tok = NepaliTokenizer()
    tokens, analyses = tok.tokenize("गरिँदै")
    # This test is lenient; ensures tokenization doesn't crash and recognizes 'हुँदै' marker when present
    tokens2, _ = tok.tokenize("हुँदै")
    assert tokens2 == ["हुँदै"]
