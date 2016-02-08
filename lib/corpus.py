from __future__ import unicode_literals
import codecs
import re
import numpy as np

num_regexp      = re.compile(r'-?[0-9]+[,.0-9]+')
meta_tag_regexp = re.compile(r'(<[^> ]+>)')
dot_guys_regexp = re.compile(r'([,.\'"?.])')
spaces_regexp   = re.compile(r'( +)')

# Allow for train and test data
train_allow_tags = ["<unk>", "<bos>", "<pad>", "<eos>", "<br>"]

def open(path):
    c = EnMarkCorpus(path)
    c.open()
    return c

class Corpus(object):
    def __init__(self, input_file):
        self.input_file = input_file
        self.rows = []
        self.vocab = {}
        self.bacov = {}
        for char in train_allow_tags:
            self.add_vocab(char)
        self.train_allow_tag_ids = self.tokens_to_ids(train_allow_tags)

    def open(self):
        with codecs.open(self.input_file) as f:
            for line in f:
                self.parse(line)

    def size(self):
        return len(self.rows)

    # parse --------------------------------------------
    def parse(self, line):
        if len(line.strip()) > 0:
            tokens = self.tokenize(line)
            self.add_row(tokens)

    def add_row(self, tokens):
        for token in tokens:
            self.add_vocab(token)
        ids = self.tokens_to_ids(tokens)
        self.rows.append(ids)

    def add_vocab(self, char):
        if not char in self.vocab:
            id = len(self.vocab)
            self.vocab[char] = id
            self.bacov[id] = char

    # encode -------------------------------------------
    def encode(self, str):
        return self.tokens_to_ids(self.tokenize(str, cleanup_tag=True))

    def tokenize(self, line):
        raise Exception("Not implemented")

    def tokens_to_ids(self, tokens):
        return [self.token_to_id(t) for t in tokens]

    def token_to_id(self, token):
        if token in self.vocab:
            return self.vocab[token]
        else:
            return self.vocab["<unk>"]

    # decode --------------------------------------------
    def decode(self, ids):
        return " ".join(self.ids_to_tokens(ids))

    def get_row(self, index):
        return self.ids_to_tokens(self.rows[index])

    def ids_to_tokens(self, ids):
        return [self.id_to_token(id) for id in ids]

    def id_to_token(self, id):
        return self.bacov[id]

    # X and Y -------------------------------------------
    def data_at(self, index):
        return [id for id in self.rows[index] if not self.is_teacher_tag(id)]

    def teacher_at(self, index):
        return [id for id in self.rows[index]]

    def is_teacher_tag(self, id):
        return self.is_meta_tag(id) and not (id in self.train_allow_tag_ids)

    def is_meta_tag(self, id):
        return re.match(meta_tag_regexp, self.id_to_token(id))

class EnMarkCorpus(Corpus):
    def __init__(self, input_file):
        super(EnMarkCorpus, self).__init__(input_file)

    def tokenize(self, line, cleanup_tag=False):
        line = line.lower()
        if cleanup_tag:
            line = re.sub(meta_tag_regexp, " ", line)
        line = self.replace_number(line)
        line = self.expand_abbr(line)
        line = self.split_text(line)
        return line

    def split_text(self, line):
        line = '<bos>' + line.replace('\n', '<br>') + '<eos>'
        line = re.sub(dot_guys_regexp, r' \1 ', line)
        line = re.sub(meta_tag_regexp, r' \1 ', line)
        line = re.sub(spaces_regexp,   r'\1',   line)
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

    def replace_number(self, line):
        return re.sub(num_regexp, '<number>', line)
