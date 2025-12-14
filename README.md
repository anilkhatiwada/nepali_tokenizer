# Nepali Grammar-Based Tokenizer

Deterministic, rule-based tokenizer for Nepali (Devanagari U+0900–U+097F). Implements sentence segmentation, word segmentation, postpositions, genitives, pluralization, particles, verb morphology (root + morphemes), pronoun inflection, sandhi rules, compounds, numerals, and mixed text handling.

## Features
- Sentence enders: `। ? ! ॥` preserved as tokens
- Word segmentation preserving punctuation
- Postpositions/case markers: ले, लाई, मा, बाट, देखि, सम्म, सँग, द्वारा
- Genitives: को/का/की/के
- Plural: हरू
- Particles: नै, पनि, त, चाहिँ
- Verb morphology: root + morphemes (e.g., `गर्दैछ → गर + दै + छ`)
- Pronoun inflection & sandhi: e.g., `उनले → उनी + ले`
- Compounds longest-match: e.g., `पुस्तकालयबाट → पुस्तकालय + बाट`
- Nepali numerals: ०-९ kept as single tokens
- English/mixed words kept intact

## Usage
```python
from nepali_tokenizer import NepaliTokenizer

text = "रामले विद्यालयमा किताब पढिरहेको थियो।"
tok = NepaliTokenizer()
flat, analyses = tok.tokenize(text, hierarchical=True)
print(flat)
# ['राम', 'ले', 'विद्यालय', 'मा', 'किताब', 'पढ', 'रहेको', 'थियो', '।']
print(analyses[5])  # TokenAnalysis for verb
```

### CLI
# Debug and profiling
```zsh
nepali-tokenize --debug --profile <<'EOF'
विद्यालयमा पढ्दै छ।
EOF
# stderr will include lines like:
# debug: WORD:विद्यालयमा
# debug: COMPOUND-EXACT:विद्यालय->विद्या+लय
# debug: CASE-SPLIT:मा->मा+
# profile: tokenize=0.000123s tokens=6
```

### Rule Catalog
- Postpositions: ले, लाई, मा, बाट, देखि, सम्म, सँग, द्वारा
- Genitives: का, की, को, के
- Plural: हरू
- Particles: नै, पनि, त, चाहिँ
- Verb morphemes (selected): ए, यो, एको, रहेको, रहेका, रहिरहेको, गइरहेको, गइरहेका, गइरहिरहेको, दै, हुँदै, छ, छन्, थियो, थिए, थे, हुँछ, हुन्छ
- Sandhi: उनी+ले→उनीले, यो+मा→योमा, यस+मा→यसमा, उन+ले→उनले
- Compounds: विद्यालय→विद्या+लय, पुस्तकालय→पुस्तक+लय, कार्यालय→कार्य+आलय, जलविद्युत→जल+विद्युत, कृषिभूमि→कृषि+भूमि, विद्युतगृह→विद्युत+गृह

Precedence: sentence → word → compounds (exact, prefix) → verb morphology → case/genitive/plural/particles → auxiliaries.

Install editable or ensure the venv is active. Then:

```zsh
# Hierarchical output (token \t root \t suffixes \t pos)
nepali-tokenize --hier <<'EOF'
विद्यालयमा पढ्दै छ।
EOF

# Flat tokens from a file
nepali-tokenize path/to/input.txt

# Structured outputs
# JSON
nepali-tokenize --format json <<'EOF'
विद्यालयमा पढ्दै छ।
EOF
# JSONL (one object per token)
nepali-tokenize --format jsonl <<'EOF'
विद्यालयमा पढ्दै छ।
EOF
# CoNLL-like (default with --hier)
nepali-tokenize --format conll --hier <<'EOF'
विद्यालयमा पढ्दै छ।
EOF

# Disable certain rules (example: no compounds, no verb splits)
nepali-tokenize --no-compounds --no-verbs --hier < input.txt

# Load external compound decompositions
cat > compounds.json <<'JSON'
{
	"काठमाडौं": ["काठ", "माण्डु"],
	"नेपालगञ्ज": ["नेपाल", "गञ्ज"]
}
JSON
nepali-tokenize --compounds compounds.json --hier <<'EOF'
नेपालगञ्जमा कार्यक्रम भयो।
EOF
```

Flags:
- `--flat`: output only tokens (default behavior)
- `--hier`: include per-token analyses aligned on lines
- `--no-sentence`: disable sentence segmentation
- `--no-case`, `--no-genitive`, `--no-plural`, `--no-particles`: disable respective suffix splits
- `--no-verbs`: disable verb morphology splitting
- `--no-pronouns`: disable pronoun POS hints
- `--no-sandhi`: disable sandhi handling
- `--no-compounds`: disable compound handling
 - `--compounds <file>`: load external compound decompositions (JSON mapping token→parts)
 - `--sandhi <file>`: load extra sandhi patterns (JSON list)
 - `--verbs <file>`: load extra verb morphemes (JSON list)

### External Rule Files
- `--compounds` expects a JSON object mapping tokens to arrays of parts:
	```json
	{"काठमाडौं": ["काठ", "माण्डु"], "नेपालगञ्ज": ["नेपाल", "गञ्ज"]}
	```
- `--sandhi` expects a JSON array. Each element can be either:
	- `["उनीले", ["उनी", "ले"]]` (fused form and its split parts), or
	- `["", "योमा", ["यो", "मा"]]` (optional original, fused, split tuple)
	Minimal form with just fused+split is recommended.
- `--verbs` expects a JSON array of morphemes to merge, e.g.:
	```json
	["गइरहेको", "हुँदै"]
	```
These external lists are merged with built-ins at runtime and respected across splitting and classification.

Example files are provided under `examples/`:
- `examples/compounds.json`
- `examples/sandhi.json`
- `examples/verbs.json`

Quick start with examples:
```zsh
nepali-tokenize --hier --format conll \
	--compounds examples/compounds.json \
	--sandhi examples/sandhi.json \
	--verbs examples/verbs.json <<'EOF'
उनीले विद्यालयमा पढ्दै छ।
EOF
```

## Tests
```bash
python -m pytest -q
```

## Benchmark
```zsh
python scripts/bench.py path/to/corpus.txt --iters 100
python scripts/bench.py path/to/corpus.txt --iters 100 --hier
```

## Evaluation
Prepare a JSONL with lines like:
```json
{"text": "विद्यालयमा पढ्दै छ।", "tokens": ["विद्यालय", "मा", "पढ", "दै", "छ", "।"]}
```
Run:
```zsh
python scripts/evaluate.py path/to/gold.jsonl
```

Optional token categories for detailed metrics:
```json
{
	"text": "विद्यालयमा पढ्दै छ।",
	"tokens": ["विद्यालय", "मा", "पढ", "दै", "छ", "।"],
	"types": ["word", "postposition", "verb_root", "verb_suffix", "verb_suffix", "punctuation"]
}
```
Whitespace baseline comparison:
```zsh
python scripts/evaluate.py path/to/gold.jsonl --baseline whitespace
```

## Notes on Ambiguity
When ambiguity arises, longest-suffix-first is applied, followed by verb root hints and explicit sandhi patterns. Rules favor linguistically accepted Nepali morphology and preserve grammatical meaning.
