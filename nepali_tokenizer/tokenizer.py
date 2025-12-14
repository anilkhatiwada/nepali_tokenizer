from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import re
from .rules import (
    DEVANAGARI_RANGE,
    NEPALI_DIGITS,
    SENTENCE_ENDERS,
    PUNCTUATION,
    POSTPOSITIONS,
    GENITIVE,
    PLURAL,
    PARTICLES,
    PRONOUN_BASE,
    VERB_MORPHEMES,
    COMPOUND_PREFIXES,
    COMPOUND_DECOMPOSITIONS,
    ROOT_HINTS,
    SANDHI_PATTERNS,
)
RE_NEPALI_NUMBER = re.compile(rf"^[{NEPALI_DIGITS}]+$")
RE_ENGLISH_WORD = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class Token:
    text: str
    type: str


@dataclass(frozen=True)
class TokenAnalysis:
    root: Optional[str]
    suffixes: List[str]
    pos: Optional[str]

def is_devanagari(s: str) -> bool:
    return all(DEVANAGARI_RANGE[0] <= ord(ch) <= DEVANAGARI_RANGE[1] for ch in s)


def longest_suffix_split(word: str, suffixes: List[str]) -> Optional[Tuple[str, str]]:
    for suf in sorted(suffixes, key=len, reverse=True):
        if word.endswith(suf) and len(word) > len(suf):
            stem = word[: -len(suf)]
            if stem:
                return stem, suf
    return None


def split_all_suffixes(word: str, tables: List[List[str]]) -> Tuple[str, List[str]]:
    stem = word
    suffixes: List[str] = []
    changed = True
    while changed:
        changed = False
        for table in tables:
            res = longest_suffix_split(stem, table)
            if res:
                stem, suf = res
                suffixes.append(suf)
                changed = True
                break
    return stem, suffixes



class RuleConfig:
    def __init__(self,
                 enable_sentence_segmentation: bool = True,
                 enable_word_segmentation: bool = True,
                 enable_postpositions: bool = True,
                 enable_genitives: bool = True,
                 enable_plural: bool = True,
                 enable_particles: bool = True,
                 enable_verb_morphology: bool = True,
                 enable_pronoun_inflection: bool = True,
                 enable_sandhi: bool = True,
                 enable_compounds: bool = True) -> None:
        self.enable_sentence_segmentation = enable_sentence_segmentation
        self.enable_word_segmentation = enable_word_segmentation
        self.enable_postpositions = enable_postpositions
        self.enable_genitives = enable_genitives
        self.enable_plural = enable_plural
        self.enable_particles = enable_particles
        self.enable_verb_morphology = enable_verb_morphology
        self.enable_pronoun_inflection = enable_pronoun_inflection
        self.enable_sandhi = enable_sandhi
        self.enable_compounds = enable_compounds


class NepaliTokenizer:
    def __init__(self, 
                 config: Optional[RuleConfig] = None, 
                 extra_compounds: Optional[dict] = None, 
                 extra_sandhi: Optional[List] = None,
                 extra_verbs: Optional[List[str]] = None,
                 debug: bool = False) -> None:
        self.config = config or RuleConfig()
        self.debug = debug
        self._trace: List[str] = []
        # Merge external compound decompositions (user-provided) with built-ins
        self.compound_decompositions = dict(COMPOUND_DECOMPOSITIONS)
        if extra_compounds:
            # Expect mapping: token -> [part1, part2, ...]
            for k, v in extra_compounds.items():
                if isinstance(v, list) and all(isinstance(x, str) for x in v):
                    self.compound_decompositions[k] = v
        # Merge external sandhi patterns: list of tuples (origA+origB, fused, (A,B)) or direct (fused, (A,B))
        self.sandhi_patterns = list(SANDHI_PATTERNS)
        if extra_sandhi:
            for item in extra_sandhi:
                if isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[0], str) and isinstance(item[1], (list, tuple)):
                    fused, split = item
                    if isinstance(split, (list, tuple)) and len(split) == 2 and all(isinstance(x, str) for x in split):
                        self.sandhi_patterns.append(("" , fused, (split[0], split[1])))
                elif isinstance(item, (list, tuple)) and len(item) == 3:
                    self.sandhi_patterns.append(tuple(item))
        # Merge external verb morphemes
        self.verb_morphemes = list(VERB_MORPHEMES)
        if extra_verbs:
            for m in extra_verbs:
                if isinstance(m, str) and m not in self.verb_morphemes:
                    self.verb_morphemes.append(m)
        # Cache sorted suffix tables to avoid sorting on every call
        self._sorted_post = sorted(POSTPOSITIONS, key=len, reverse=True)
        self._sorted_gen = sorted(GENITIVE, key=len, reverse=True)
        self._sorted_plural = sorted(PLURAL, key=len, reverse=True)
        self._sorted_particles = sorted(PARTICLES, key=len, reverse=True)
        self._sorted_verbs = sorted(self.verb_morphemes, key=len, reverse=True)

    def segment_sentences(self, text: str) -> List[str]:
        if not self.config.enable_sentence_segmentation:
            return [text]
        parts: List[str] = []
        buf: List[str] = []
        for ch in text:
            buf.append(ch)
            if ch in SENTENCE_ENDERS:
                parts.append("".join(buf).strip())
                buf = []
        if buf:
            remaining = "".join(buf).strip()
            if remaining:
                parts.append(remaining)
        return parts

    def segment_words(self, sentence: str) -> List[Token]:
        tokens: List[Token] = []
        i = 0
        while i < len(sentence):
            ch = sentence[i]
            if ch.isspace():
                i += 1
                continue
            if ch in SENTENCE_ENDERS:
                tokens.append(Token(ch, "punctuation"))
                i += 1
                continue
            if ch in PUNCTUATION:
                tokens.append(Token(ch, "punctuation"))
                i += 1
                continue
            j = i
            while j < len(sentence):
                cj = sentence[j]
                if cj.isspace() or cj in SENTENCE_ENDERS or cj in PUNCTUATION:
                    break
                j += 1
            word = sentence[i:j]
            if RE_NEPALI_NUMBER.match(word):
                tokens.append(Token(word, "numeral"))
            elif RE_ENGLISH_WORD.match(word):
                tokens.append(Token(word, "english"))
            else:
                tokens.append(Token(word, "word"))
            i = j
        return tokens

    def analyze_word(self, word: str) -> Tuple[List[Token], TokenAnalysis]:
        if self.debug:
            self._trace.append(f"WORD:{word}")
        if self.config.enable_sandhi:
            for _orig, fused, split in self.sandhi_patterns:
                if word == fused:
                    if self.debug:
                        self._trace.append(f"SANDHI:{fused}->{split[0]}+{split[1]}")
                    a, b = split
                    t1, _ = self.analyze_word(a)
                    t2, _ = self.analyze_word(b)
                    return t1 + t2, TokenAnalysis(root=None, suffixes=[], pos=None)

        if RE_ENGLISH_WORD.match(word):
            if self.debug:
                self._trace.append("ENGLISH")
            return [Token(word, "english")], TokenAnalysis(root=None, suffixes=[], pos=None)

        if RE_NEPALI_NUMBER.match(word):
            if self.debug:
                self._trace.append("NUMERAL")
            return [Token(word, "numeral")], TokenAnalysis(root=None, suffixes=[], pos=None)

        # Treat standalone auxiliaries as atomic verb tokens
        AUX_ATOMIC = {"छ", "छन्", "थियो", "थिए", "हुँछ", "छैन", "हुन्छ", "रहेछ", "हुँदै"}
        if word in AUX_ATOMIC:
            if self.debug:
                self._trace.append(f"AUX:{word}")
            return [Token(word, "verb_suffix")], TokenAnalysis(root=None, suffixes=[word], pos=None)

        if self.config.enable_compounds:
            # Exact word decomposition first
            if word in self.compound_decompositions:
                parts = self.compound_decompositions[word]
                if self.debug:
                    self._trace.append(f"COMPOUND-EXACT:{word}->{'+'.join(parts)}")
                toks: List[Token] = []
                for p in parts:
                    toks.append(Token(p, "compound"))
                return toks, TokenAnalysis(root=None, suffixes=[], pos=None)
            for pref in sorted(COMPOUND_PREFIXES, key=len, reverse=True):
                if word.startswith(pref) and len(word) > len(pref):
                    rest = word[len(pref):]
                    if self.debug:
                        self._trace.append(f"COMPOUND-PREFIX:{pref}|REST:{rest}")
                    stem2, suff2 = split_all_suffixes(rest, [self._sorted_post, self._sorted_gen, self._sorted_plural, self._sorted_particles])
                    # If prefix has a decomposition, emit its parts; otherwise emit prefix
                    if pref in self.compound_decompositions:
                        toks = [Token(p, "compound") for p in self.compound_decompositions[pref]]
                    else:
                        toks = [Token(pref, "compound")]
                    if stem2:
                        toks.append(Token(stem2, "word"))
                    for s in suff2:
                        toks.append(Token(s, self.classify_suffix(s)))
                    return toks, TokenAnalysis(root=None, suffixes=[], pos=None)

        # Prefer verb morphology first to avoid splitting 'रहेको' → 'को'
        verb_root = None
        verb_suffixes: List[str] = []
        stem_for_case = word
        # Heuristic: attempt verb split only if clear markers or known roots
        has_marker = any(word.endswith(m) for m in VERB_MORPHEMES)
        has_marker = any(word.endswith(m) for m in self.verb_morphemes)
        starts_with_root = any(word.startswith(r) for r in ROOT_HINTS)
        if self.config.enable_verb_morphology and is_devanagari(word) and (has_marker or starts_with_root):
            s2, vsufs = split_all_suffixes(word, [self._sorted_verbs])
            if s2 != word:
                # Normalize stem by removing trailing 'ि' (U+093F) and virama '्' (U+094D)
                stem_norm = s2.rstrip("\u093F\u094D")
                verb_root = stem_norm
                verb_suffixes = vsufs
                if self.debug:
                    self._trace.append(f"VERB-SPLIT:{word}->{verb_root}+{'+'.join(verb_suffixes) if verb_suffixes else ''}")
                stem_for_case = verb_root  # case markers, if any, apply after verb split
            else:
                for r in sorted(ROOT_HINTS, key=len, reverse=True):
                    if word.startswith(r) and len(word) > len(r):
                        verb_root = r
                        remainder = word[len(r):]
                        if remainder:
                            # Keep imperative forms intact (e.g., 'देऊ')
                            if remainder == "ऊ":
                                if self.debug:
                                    self._trace.append("VERB-IMPERATIVE-KEEP")
                                verb_root = None
                                verb_suffixes = []
                                stem_for_case = word
                                break
                            # Infinitive 'नु' handling: खानु → खा + नु
                            if remainder == "नु":
                                if self.debug:
                                    self._trace.append("VERB-INF-NOON:नु")
                                verb_suffixes = ["नु"]
                                stem_for_case = verb_root
                                break
                            # If remainder begins with dependent vowel marks, skip them for suffix detection
                            # Skip leading dependent vowels except 'े' to allow past 'ए' mapping
                            dependent_vowels = set("िीुूोैौा")
                            k = 0
                            while k < len(remainder) and remainder[k] in dependent_vowels:
                                k += 1
                            rem2 = remainder[k:] if k else remainder
                            # Map matra 'े' at end to 'ए' morpheme
                            if rem2.endswith('े'):
                                verb_suffixes = ['ए']
                                if self.debug:
                                    self._trace.append("VERB-PAST:ए")
                            else:
                                _, vsufs2 = split_all_suffixes(rem2, [VERB_MORPHEMES])
                                verb_suffixes = vsufs2
                        stem_for_case = verb_root
                        break

        tokens: List[Token] = []
        analysis_suffixes: List[str] = []

        if verb_root:
            # Normalize trailing virama (्) in standalone root
            root_out = verb_root[:-1] if verb_root and verb_root.endswith("\u094D") else verb_root
            tokens.append(Token(root_out, "verb_root"))
            for s in verb_suffixes:
                tokens.append(Token(s, "verb_suffix"))
                analysis_suffixes.append(s)
        else:
            # No verb split; proceed with case markers on the whole word
            stem_for_case = word

        # Now apply case markers, genitives, plural, particles on remaining stem, respecting config toggles
        suffix_tables: List[List[str]] = []
        if self.config.enable_postpositions:
            suffix_tables.append(POSTPOSITIONS)
        if self.config.enable_genitives:
            suffix_tables.append(GENITIVE)
        if self.config.enable_plural:
            suffix_tables.append(PLURAL)
        if self.config.enable_particles:
            suffix_tables.append(PARTICLES)

        if suffix_tables:
            stem, suffixes = split_all_suffixes(stem_for_case, suffix_tables)
            if self.debug:
                self._trace.append(f"CASE-SPLIT:{stem_for_case}->{stem}+{'+'.join(suffixes) if suffixes else ''}")
        else:
            stem, suffixes = stem_for_case, []
        if not verb_root and stem:
            tokens.append(Token(stem, "word"))

        for s in suffixes:
            ttype = self.classify_suffix(s)
            tokens.append(Token(s, ttype))
            analysis_suffixes.append(s)

        pos_hint = None
        if self.config.enable_pronoun_inflection and stem in PRONOUN_BASE:
            pos_hint = "pronoun"

        return tokens, TokenAnalysis(root=verb_root or (stem if stem else None), suffixes=analysis_suffixes, pos=pos_hint)

    def get_trace(self) -> List[str]:
        return self._trace
    

    def classify_suffix(self, s: str) -> str:
        if s in POSTPOSITIONS:
            return "postposition"
        if s in GENITIVE:
            return "genitive"
        if s in PLURAL:
            return "plural"
        if s in PARTICLES:
            return "particle"
        if s in self.verb_morphemes:
            return "verb_suffix"
        return "suffix"

    def tokenize(self, text: str, hierarchical: bool = True) -> Tuple[List[str], Optional[List[TokenAnalysis]]]:
        tokens_flat: List[str] = []
        analyses: List[TokenAnalysis] = []
        for sentence in self.segment_sentences(text):
            raw_tokens = self.segment_words(sentence)
            for tk in raw_tokens:
                if tk.type in {"punctuation", "numeral", "english"}:
                    tokens_flat.append(tk.text)
                    analyses.append(TokenAnalysis(root=None, suffixes=[], pos=None))
                    continue
                analyzed, analysis = self.analyze_word(tk.text)
                for a in analyzed:
                    tokens_flat.append(a.text)
                analyses.append(analysis)
        return tokens_flat, (analyses if hierarchical else None)

    def tokenize_stream(self, texts: List[str], hierarchical: bool = True) -> Tuple[List[str], Optional[List[TokenAnalysis]]]:
        tokens_flat: List[str] = []
        analyses_all: List[TokenAnalysis] = []
        for text in texts:
            flat, analyses = self.tokenize(text, hierarchical=hierarchical)
            tokens_flat.extend(flat)
            if hierarchical and analyses is not None:
                analyses_all.extend(analyses)
        return tokens_flat, (analyses_all if hierarchical else None)


__all__ = ["NepaliTokenizer", "RuleConfig", "Token", "TokenAnalysis"]