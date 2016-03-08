from __future__ import unicode_literals
import pytest
import sys
sys.path.append('lib')

import corpus
import mark

test_file = "tests/test.html"

def pytest_funcarg__test_corp(request):
    return corpus.open(test_file)

def test_idx_to_type():
    assert mark.idx_to_type(1) == '<sj>'

def test_convert_teach_id_row(test_corp):
    tokens   = ["<bos>", "<sj>", "james", "</sj>", "<v>", "is", "</v>", "<eos>"]
    mark_vec = mark.convert_teach_id_row(test_corp.tokens_to_ids(tokens), test_corp)
    assert (mark_vec[0] == mark.convert_types_to_vec([])).all()
    assert (mark_vec[1] == mark.convert_types_to_vec(['<sj>'])).all()
    assert (mark_vec[2] == mark.convert_types_to_vec(['<v>'])).all()
    assert (mark_vec[3] == mark.convert_types_to_vec([])).all()

    tokens   = ["<bos>", "<sj>", "A", "<v>", "B", "</v>", "</sj>", "<eos>"]
    mark_vec = mark.convert_teach_id_row(test_corp.tokens_to_ids(tokens), test_corp)
    assert (mark_vec[0] == mark.convert_types_to_vec([])).all()
    assert (mark_vec[1] == mark.convert_types_to_vec(['<sj>'])).all()
    assert (mark_vec[2] == mark.convert_types_to_vec(['<sj>', '<v>'])).all()
    assert (mark_vec[3] == mark.convert_types_to_vec([])).all()

def test_convert_types_to_vec(test_corp):
    vec = mark.convert_types_to_vec(['<sj>', '<v>'])
    assert vec.shape == (mark.mark_dim_size(),)
    assert vec[0] == 0.0
    assert vec[1] == 1.0
    assert vec[2] == 1.0
    assert vec[3] == 0.0
    assert vec[4] == 0.0
