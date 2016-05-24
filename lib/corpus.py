from __future__ import unicode_literals
import codecs
import yaml
import re
import numpy as np
from nltk.translate.bleu_score import bleu
from text_tagger import DummyPosTagger, NERTagger

num_regexp      = re.compile(r'-?[0-9]+[,.0-9]+')
meta_tag_regexp = re.compile(r'(<[^> ]+>)')
dot_guys_regexp = re.compile(r'([,.\'"?.])')
spaces_regexp   = re.compile(r'( +)')

# Allow for train and test data
train_allow_tags = ["<unk>", "<bos>", "<pad>", "<eos>", "<br>", "<number>"]

def dummy_tagger():
    return DummyPosTagger()

def tagger():
    return NERTagger()

def open(path, tagger=dummy_tagger()):
    c = EnMarkCorpus(path, tagger=tagger)
    c.open()
    return c

class Corpus(object):
    def __init__(self, input_file=None):
        self.input_file = input_file
        self.rows = []
        self.pos_rows = []
        self.vocab = {}
        self.bacov = {}
        self.frequency = {}
        self.minor_word_frequency = 1
        for char in train_allow_tags:
            self.add_vocab(char)
        self.train_allow_tag_ids = self.tokens_to_ids(train_allow_tags)

    def __eq__(self, other):
        return (self.rows  == other.rows) and \
               (self.pos_rows  == other.pos_rows) and \
               (self.vocab == other.vocab) and \
               (self.bacov == other.bacov)

    def __ne__(self, other):
        return not self == other

    def save(self, file_path):
        data = {
            'rows':  self.rows,
            'pos':   self.pos_rows,
            'vocab': self.vocab,
            'bacov': self.bacov
        }
        with codecs.open(file_path, 'w') as file:
            yaml.dump(data, file, encoding='utf8', allow_unicode=True)

    @staticmethod
    def load(file_path):
        corpus = Corpus()
        corpus.deserialize(file_path)
        return corpus

    def deserialize(self, file_path):
        with codecs.open(file_path, 'r') as file:
            data = yaml.load(file.read())
            self.merge(data)

    def merge(self, new_data):
        self.rows = new_data['rows']
        self.pos_rows = new_data['pos']
        self.vocab = new_data['vocab']
        self.bacov = new_data['bacov']

    def open(self):
        with codecs.open(self.input_file) as f:
            for line in f:
                self.parse(line)

    def size(self):
        return len(self.rows)

    def vocab_size(self):
        return len(self.vocab)

    # parse --------------------------------------------
    def parse(self, line):
        if len(line.strip()) > 0:
            tokens = self.tokenize(line)
            self.add_row(self.rows, tokens)

    def add_row(self, rows, tokens):
        for token in tokens:
            self.add_vocab(token)
        ids = self.tokens_to_ids(tokens)
        rows.append(ids)

    def add_vocab(self, char):
        if not char in self.vocab:
            id = len(self.vocab)
            self.vocab[char] = id
            self.bacov[id] = char
            self.frequency[id] = 1
        else:
            id = self.vocab[char]
            self.frequency[id] += 1

    # encode -------------------------------------------
    def encode(self, str):
        return self.tokens_to_ids(self.tokenize(str, cleanup_tag=True))

    def tokenize(self, line):
        raise Exception("Not implemented")

    def tokens_to_ids(self, tokens):
        return [self.token_to_id(t) for t in tokens]

    def token_to_id(self, token):
        if self.known_word(token):
            return self.vocab[token]
        else:
            return self.vocab["<unk>"]

    def known_word(self, token):
        return token in self.vocab

    # decode --------------------------------------------
    def decode(self, ids):
        return " ".join(self.ids_to_tokens(ids))

    def get_row(self, index):
        return self.ids_to_tokens(self.rows[index])

    def ids_to_tokens(self, ids):
        return [self.id_to_token(id) for id in ids]

    def id_to_token(self, id):
        return self.bacov[id]

    # X vector -------------------------------------------
    def data_at(self, index):
        return [self.convert_minor_word(id, word_idx, index)
                for word_idx, id
                in enumerate(self.rows[index])
                if not self.is_teacher_tag(id)]

    def convert_minor_word(self, id, word_index, row_index):
        if self.is_minor_word(id):
            return self.token_to_id("<unk>")
        else:
            return id

    def is_minor_word(self, id):
        return self.frequency[id] <= self.minor_word_frequency

    # Y vector --------------------------------------------
    def teacher_at(self, index):
        return [id
                for word_idx, id
                in enumerate(self.rows[index])]

    def is_teacher_tag(self, id):
        return self.is_meta_tag_id(id) and not (id in self.train_allow_tag_ids)

    def is_meta_tag_id(self, id):
        return self.is_meta_tag(self.id_to_token(id))

    def is_meta_tag(self, token):
        return re.match(meta_tag_regexp, token)

    def bleu_score(self, candidate, references):
        weights    = [0.5, 0.5]
        candidate  = [c for c in candidate if c != '<pad>']
        references = [[c for c in ref if c != '<pad>'] for ref in references]
        return bleu(references, candidate, weights)

class EnMarkCorpus(Corpus):
    def __init__(self, input_file=None, tagger=None):
        super(EnMarkCorpus, self).__init__(input_file)
        self.tagger = tagger

    @staticmethod
    def load(file_path):
        corpus = EnMarkCorpus()
        corpus.deserialize(file_path)
        return corpus

    def parse(self, line):
        if len(line.strip()) > 0:
            tokens = self.tokenize(line)
            self.add_row(self.rows, tokens)
            tags = self.pos_tag(tokens)
            self.add_row(self.pos_rows, tags)

    def tokenize(self, line, cleanup_tag=False):
        line = line.lower()
        if cleanup_tag:
            line = re.sub(meta_tag_regexp, " ", line)
        line = self.replace_number(line)
        line = self.expand_abbr(line)
        line = self.split_text(line)
        return line

    def pos_tag(self, tokens):
        pos_safe_tokens = []
        for token in tokens:
            if not self.is_meta_tag(token):
                pos_safe_tokens.append(token)
        idx = 0
        pos_tags = ["<POS:META>"] * len(tokens)
        tagged_tokens = self.tagger.tag(pos_safe_tokens)
        for word, tag in tagged_tokens:
            while not word == tokens[idx]:
                idx += 1
            pos_tags[idx] = "<POS:{0}>".format(tag)
        return pos_tags

    def convert_minor_word(self, id, word_index, row_index):
        if self.is_minor_word(id):
            return self.pos_tag_at(row_index)[word_index]
        else:
            return id

    def pos_tag_at(self, index):
        return self.pos_rows[index]

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
