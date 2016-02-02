from __future__ import unicode_literals
import codecs
import re

def open(path):
    c = EnMarkCorpus(path)
    c.open()
    return c

class Corpus(object):
    def __init__(self, input_file):
        self.input_file = input_file
        self.rows = []
        self.vocab      = {}
        for char in ["<unk>", "<bol>", "<pad>", "<eol>"]:
            self.add_vocab(char)

    def open(self):
        with codecs.open(self.input_file) as f:
            for line in f:
                self.parse(line)

    def parse(self, line):
        raise Exception("Not implemented")

    def add_row(self, row):
        for token in row:
            self.add_vocab(token)
        self.rows.append(row)

    def add_vocab(self, char):
        if not char in self.vocab:
            self.vocab[char] = len(self.vocab)

class EnMarkCorpus(Corpus):
    def __init__(self, input_file):
        super(EnMarkCorpus, self).__init__(input_file)
        for char in ["<s>", "<ss>"]:
            self.add_vocab(char)

    def parse(self, line):
        if len(line.strip()) > 0:
            line = line.lower()
            line = self.expand_abbr(line)
            line = self.split_text(line)
            self.add_row(line)

    def split_text(self, line):
        line = '<bol>' + line.replace('\n', '<eol>')
        line = re.sub(r'([,.\'"?.])', r" \1 ", line)
        line = re.sub(r'(<[^>]+>)', r" \1 ", line)
        line = re.sub(r'( +)', r"\1", line)
        return line.split()

    def expand_abbr(self, line):
        table = {"haven't": "have not",
                 "hasn't":  "has not",
                 "what's":  "what is",
                 "it's":    "it is",
                 "i'm":     "i am"}
        for b, a in table.items():
            line = line.replace(b, a)
        return line
