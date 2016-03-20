from __future__ import unicode_literals
import codecs
import numpy as np
from nltk.tag.perceptron import PerceptronTagger

class DummyPosTagger:
    def tag(self, tokens):
        return [(token, "DUMMY") for token in tokens]

class NERTagger:
    def __init__(self):
        self.tagger = PerceptronTagger()

    def tag(self, tokens):
        return self.tagger.tag(tokens)
