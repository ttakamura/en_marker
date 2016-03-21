from __future__ import unicode_literals
import pytest
import numpy as np
import sys
sys.path.append('lib')

import corpus
from minbatch import MinBatch
from minbatch import MarkTeacherMinBatch
import mark
import config

np.random.seed(123)

test_file = "tests/test.html"

def pytest_funcarg__test_corp(request):
    return corpus.open(test_file, tagger=corpus.dummy_tagger())

def pytest_funcarg__pos_tag_corp(request):
    return corpus.open(test_file, tagger=corpus.tagger())

def pytest_funcarg__test_conf(request):
    args = "--mode train".split(" ")
    return config.parse_args(raw_args = args)

# ------- test ------------------------
def test_init_corpus():
    c = corpus.EnMarkCorpus(test_file, tagger=corpus.dummy_tagger())
    assert c.vocab['<unk>'] == 0

def test_size(test_corp):
    assert 5 < test_corp.size()

def test_parse(test_corp):
    assert test_corp.get_row(0) == ["<bos>", "<sj>", "james", "</sj>", "<v>", "is", "</v>", "a", "teacher", ".", "<br>", "<eos>"]

def test_parse_abbrev(test_corp):
    assert test_corp.get_row(1) == ["<bos>", "i", "am", "tom", ".", "<br>", "<eos>"]
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
    assert test_corp.tokenize("<sj>James</sj> is.") == ["<bos>", "<sj>", "james", "</sj>", "is", ".", "<eos>"]
    assert test_corp.tokenize("<sj>James</sj> is.", cleanup_tag=True) == ["<bos>", "james", "is", ".", "<eos>"]
    assert test_corp.tokenize("3 > 1", cleanup_tag=True) == ["<bos>", "3", ">", "1", "<eos>"]

def test_encode(test_corp):
    assert [1, 6, 9, 13, 3] == test_corp.encode("james is.")

def test_decode(test_corp):
    assert "<bos> james is . <eos>" == test_corp.decode([1, 6, 9, 13, 3])

def test_data_at(test_corp):
    assert test_corp.ids_to_tokens(test_corp.data_at(0)) == ["<bos>", "<POS:DUMMY>", "is", "a", "<POS:DUMMY>", ".", "<br>", "<eos>"]

def test_teacher_at(test_corp):
    assert test_corp.ids_to_tokens(test_corp.teacher_at(0)) == ["<bos>", "<sj>", "james", "</sj>", "<v>", "is", "</v>", "a", "teacher", ".", "<br>", "<eos>"]

def test_unknown_word(test_corp):
    assert test_corp.ids_to_tokens(test_corp.encode("isetan")) == ["<bos>", "<unk>", "<eos>"]

def test_pos_tag(test_corp):
    tokens = ["james", "</sj>", "<v>", "is", "</v>", "teacher"]
    tags   = test_corp.pos_tag(tokens)
    assert tags == ["<POS:DUMMY>", "<POS:META>", "<POS:META>", "<POS:DUMMY>", "<POS:META>", "<POS:DUMMY>"]

def test_minbatch_randomized_from_corpus(test_conf, test_corp):
    train_idxs, test_idxs, trains, tests = MinBatch.randomized_from_corpus(test_conf, test_corp, 2)
    assert train_idxs.shape == (4, 2)
    assert test_idxs.shape  == (2, 2)

    train_idxs2, test_idxs2, _, _ = MinBatch.randomized_from_corpus(test_conf, test_corp, 2)
    for i in test_idxs2.reshape(4):
        for j in train_idxs.reshape(8):
            assert i != j

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

    # <sj>James</sj> <v>is</v> a teacher.
    # I haven't
    assert f(tests[0].teach_batch_at(1)) == ["<sj>",     "i"]
    assert f(tests[0].teach_batch_at(2)) == ["james", "have"]
    assert f(tests[0].teach_batch_at(7)) == ["a",       "<pad>"]
    assert f(tests[0].teach_batch_at(8)) == ["teacher", "<pad>"]

def test_save_and_load(test_corp):
    test_corp.save("./tmp/test_corp.vocab")
    new_corp = corpus.EnMarkCorpus.load("./tmp/test_corp.vocab")
    assert test_corp == new_corp

def test_bleu_score(test_corp):
    candidate  = ['this', 'is', 'a', 'pen', '<pad>', '<pad>', '<pad>', '<pad>']
    references = [['this', 'is', 'a', 'pen', '<pad>']]
    score = test_corp.bleu_score(candidate, references)
    assert score == 1.0

def test_is_minor_word(test_corp):
    assert test_corp.is_minor_word(test_corp.token_to_id("i"))     == False
    assert test_corp.is_minor_word(test_corp.token_to_id("have"))  == False
    assert test_corp.is_minor_word(test_corp.token_to_id("tom"))   == True
    assert test_corp.is_minor_word(test_corp.token_to_id("james")) == True
    assert test_corp.is_minor_word(test_corp.token_to_id("fumi"))  == True

def test_convert_minor_word(pos_tag_corp):
    test_corp = pos_tag_corp
    idx   = 2
    james = test_corp.get_row(0)[idx]
    assert james == "james"
    result = test_corp.convert_minor_word(test_corp.token_to_id("james"), idx, 0)
    assert test_corp.id_to_token(result) == "<POS:NNS>"

def test_mark_teach_minbatch(test_conf, test_corp):
    data_rows  = [test_corp.tokens_to_ids([        "i",                 "am"        ]), test_corp.tokens_to_ids([        "i"         ])]
    teach_rows = [test_corp.tokens_to_ids(["<sj>", "i", "</sj>", "<v>", "am", "</v>"]), test_corp.tokens_to_ids(["<sj>", "i", "</sj>"])]
    batch = MarkTeacherMinBatch(test_conf, test_corp, data_rows, teach_rows)
    f = lambda x: test_corp.ids_to_tokens(list(x))

    assert f(batch.data_batch_at(0)) == ["i",  "i"]
    assert f(batch.data_batch_at(1)) == ["am", "<pad>"]
    assert (batch.teach_batch_at(0)[0] == mark.convert_types_to_vec(['<sj>'])).all()
    assert (batch.teach_batch_at(0)[1] == mark.convert_types_to_vec(['<sj>'])).all()
    assert (batch.teach_batch_at(1)[0] == mark.convert_types_to_vec(['<v>'])).all()
    assert batch.teach_batch_at(1)[1]  == -1
