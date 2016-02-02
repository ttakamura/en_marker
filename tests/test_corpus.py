import pytest
import sys
sys.path.append('lib')

import corpus

test_file = "tests/test.html"

def pytest_funcarg__test_corp(request):
    return corpus.open(test_file)

def test_init_corpus():
    c = corpus.EnMarkCorpus(test_file)
    assert c.vocab['<unk>'] == 0

def test_parse(test_corp):
    assert test_corp.rows[0] == ["<bol>", "<s>", "james", "</s>", "<v>", "is", "</v>", "a", "teacher", ".", "<eol>"]
    assert test_corp.rows[1] == ["<bol>", "i", "am", "james", ".", "<eol>"]
    assert test_corp.rows[2] == ["<bol>", "i", "have", "not", "<eol>"]
    assert test_corp.rows[3] == ["<bol>", "he", "has", "not", "<eol>"]
    assert test_corp.rows[4] == ["<bol>", "what", "is", "up", "<eol>"]

def test_vocab(test_corp):
    assert "james" in test_corp.vocab
