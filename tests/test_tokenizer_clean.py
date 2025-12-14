import unittest
from nepali_tokenizer import NepaliTokenizer

class TestNepaliTokenizerClean(unittest.TestCase):
    def setUp(self):
        self.tok = NepaliTokenizer()

    def assertTokens(self, text, expected):
        tokens, _ = self.tok.tokenize(text, hierarchical=False)
        self.assertEqual(tokens, expected)

    def test_sentence_segmentation_and_punct(self):
        self.assertTokens("उसले नै गर्यो।", ["उस", "ले", "नै", "गर", "यो", "।"])  # particle preserved

    def test_postpositions(self):
        self.assertTokens(
            "रामले विद्यालयमा किताब पढिरहेको थियो।",
            ['राम', 'ले', 'विद्या', 'लय', 'मा', 'किताब', 'पढ', 'रहेको', 'थियो', '।']
        )

    def test_compound_office(self):
        self.assertTokens("कार्यालयमा।", ["कार्य", "आलय", "मा", "।"])  

    def test_infinitive(self):
        self.assertTokens("खानु।", ["खा", "नु", "।"])  

    def test_negation_aux(self):
        self.assertTokens("छैन।", ["छैन", "।"])  

    def test_honorific_pronoun(self):
        tokens, analyses = self.tok.tokenize("उहाँलाई।")
        self.assertEqual(tokens, ["उहाँ", "लाई", "।"])  
        self.assertEqual(analyses[0].pos, "pronoun")

    def test_progressive_reading(self):
        # Ensure 'पढिरहेको' normalizes root to 'पढ' not 'पढि'
        self.assertTokens("पढिरहेको।", ["पढ", "रहेको", "।"])  

    def test_plural(self):
        self.assertTokens("केटाहरू आए।", ["केटा", "हरू", "आ", "ए", "।"])  # verb split into root + morpheme

    def test_pronoun_inflection(self):
        tokens, analyses = self.tok.tokenize("मलाई किताब देऊ।")
        self.assertEqual(tokens, ["म", "लाई", "किताब", "देऊ", "।"])  # keep imperative intact
        self.assertEqual(analyses[0].pos, "pronoun")

    def test_sandhi(self):
        self.assertTokens("उनले किताब पढे।", ["उनी", "ले", "किताब", "पढ", "ए", "।"])  # verb morphology

    def test_english_and_mixed(self):
        self.assertTokens("यो book राम्रो छ।", ["यो", "book", "राम्रो", "छ", "।"])  

    def test_digits(self):
        self.assertTokens("१२३ किताब", ["१२३", "किताब"]) 

if __name__ == '__main__':
    unittest.main()
