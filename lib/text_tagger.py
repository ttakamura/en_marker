from __future__ import unicode_literals
import codecs
import numpy as np
from nltk.tag.perceptron import PerceptronTagger
import nltk

class DummyPosTagger:
    def tag(self, tokens):
        return [(token, "DUMMY") for token in tokens]

class NERTagger:
    def __init__(self):
        self.pos_tagger = PerceptronTagger()

    def tag(self, tokens):
        tree = nltk.ne_chunk(self.pos_tagger.tag(tokens))
        tagged_tokens = []
        for t in tree:
            if type(t) == nltk.tree.Tree:
                label = t.label()
                for token in t:
                    tagged_tokens.append((token[0], label))
            else:
                tagged_tokens.append(t)
        return tagged_tokens
