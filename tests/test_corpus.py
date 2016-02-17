from __future__ import unicode_literals
import pytest
import numpy as np
import sys
sys.path.append('lib')

import corpus
from corpus import MinBatch
import config

np.random.seed(123)

test_file = "tests/test.html"

def pytest_funcarg__test_corp(request):
    return corpus.open(test_file, tagger=corpus.DummyPosTagger())

def pytest_funcarg__test_conf(request):
    args = "--mode train".split(" ")
    return config.parse_args(raw_args = args)

# ------- test ------------------------
def test_init_corpus():
    c = corpus.EnMarkCorpus(test_file, tagger=corpus.DummyPosTagger())
    assert c.vocab['<unk>'] == 0

def test_size(test_corp):
    assert 5 < test_corp.size()

def test_parse(test_corp):
    assert test_corp.get_row(0) == ["<bos>", "<s>", "james", "</s>", "<v>", "is", "</v>", "a", "teacher", ".", "<br>", "<eos>"]

def test_parse_abbrev(test_corp):
    assert test_corp.get_row(1) == ["<bos>", "i", "am", "james", ".", "<br>", "<eos>"]
    assert test_corp.get_row(2) == ["<bos>", "i", "have", "not", "<br>", "<eos>"]
    assert test_corp.get_row(3) == ["<bos>", "he", "has", "not", "<br>", "<eos>"]
    assert test_corp.get_row(4) == ["<bos>", "what", "is", "up", "<br>", "<eos>"]

def test_parse_number(test_corp):
    assert test_corp.get_row(5) == ["<bos>", "<number>", "yen", ".", "<br>", "<eos>"]

def test_vocab(test_corp):
    assert "james" in test_corp.vocab

def test_add_vocab(test_corp):
    num = len(test_corp.vocab)
    test_corp.add_vocab('hello_world_this_is_dummy')
    assert test_corp.vocab['hello_world_this_is_dummy'] == num

def test_tokenize(test_corp):
    assert test_corp.tokenize("<s>James</s> is.") == ["<bos>", "<s>", "james", "</s>", "is", ".", "<eos>"]
    assert test_corp.tokenize("<s>James</s> is.", cleanup_tag=True) == ["<bos>", "james", "is", ".", "<eos>"]
    assert test_corp.tokenize("3 > 1", cleanup_tag=True) == ["<bos>", "3", ">", "1", "<eos>"]

def test_encode(test_corp):
    assert [1, 6, 9, 13, 3] == test_corp.encode("james is.")

def test_decode(test_corp):
    assert "<bos> james is . <eos>" == test_corp.decode([1, 6, 9, 13, 3])

def test_data_at(test_corp):
    assert test_corp.ids_to_tokens(test_corp.data_at(0)) == ["<bos>", "james", "is", "a", "teacher", ".", "<br>", "<eos>"]

def test_teacher_at(test_corp):
    assert test_corp.ids_to_tokens(test_corp.teacher_at(0)) == ["<bos>", "<s>", "james", "</s>", "<v>", "is", "</v>", "a", "teacher", ".", "<br>", "<eos>"]

def test_unknown_word(test_corp):
    assert test_corp.ids_to_tokens(test_corp.encode("isetan")) == ["<bos>", "<unk>", "<eos>"]

def test_pos_tag(test_corp):
    tokens = ["james", "</s>", "<v>", "is", "</v>", "teacher"]
    tags   = test_corp.pos_tag(tokens)
    assert tags == ["<POS:NN>", "<POS:META>", "<POS:META>", "<POS:NN>", "<POS:META>", "<POS:NN>"]

def test_minbatch_randomized_from_corpus(test_conf, test_corp):
    train_idxs, test_idxs, trains, tests = MinBatch.randomized_from_corpus(test_conf, test_corp, 2)
    assert train_idxs.shape == (2, 2)
    assert test_idxs.shape  == (1, 2)

def test_minbatch_from_corpus(test_conf, test_corp):
    train_idxs = [[1, 3]]
    test_idxs  = [[0, 2]]
    trains     = MinBatch.from_corpus(test_conf, test_corp, train_idxs)
    tests      = MinBatch.from_corpus(test_conf, test_corp, test_idxs)
    f = lambda x: test_corp.ids_to_tokens(list(x))

    # I'm James.
    # He hasn't
    assert f(trains[0].data_batch_at(1)) == ["i",  "he"]
    assert f(trains[0].data_batch_at(2)) == ["am", "has"]

    # <s>James</s> <v>is</v> a teacher.
    # I haven't
    assert f(tests[0].teach_batch_at(1)) == ["<s>",     "i"]
    assert f(tests[0].teach_batch_at(2)) == ["james",   "have"]
    assert f(tests[0].teach_batch_at(7)) == ["a",       "<pad>"]
    assert f(tests[0].teach_batch_at(8)) == ["teacher", "<pad>"]
