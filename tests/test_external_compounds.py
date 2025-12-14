import json
import os

from nepali_tokenizer.tokenizer import NepaliTokenizer


def test_external_compounds_simple(tmp_path):
    data = {
        "काठमाडौं": ["काठ", "माण्डु"],
        "नेपालगञ्ज": ["नेपाल", "गञ्ज"],
    }
    p = tmp_path / "compounds.json"
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    t = NepaliTokenizer(extra_compounds=data)
    tokens, _ = t.tokenize("काठमाडौं")
    assert tokens == ["काठ", "माण्डु"]
    tokens2, _ = t.tokenize("नेपालगञ्ज")
    assert tokens2 == ["नेपाल", "गञ्ज"]
