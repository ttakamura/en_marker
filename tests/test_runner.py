from __future__ import unicode_literals
import pytest
import sys
sys.path.append('lib')

import config
import corpus
from minbatch import MinBatch
import runner
from encdec import EncoderDecoder

test_file = "tests/test.html"

def pytest_funcarg__test_conf_v1(request):
    args = "--mode train --embed 50 --hidden 30 --minbatch 2 --model v1".split(" ")
    conf = config.parse_args(raw_args = args)
    conf.corpus = corpus.open(test_file)
    return conf

def pytest_funcarg__test_conf_v2(request):
    args = "--mode train --embed 50 --hidden 30 --minbatch 2 --model v2".split(" ")
    conf = config.parse_args(raw_args = args)
    conf.corpus = corpus.open(test_file)
    return conf

# -------------------- v1 -----------------------------------------------------------
def test_forward_v1(test_conf_v1):
    conf = test_conf_v1
    encdec = EncoderDecoder(conf)
    train_idxs, test_idxs, trains, tests = encdec.minbatch_class.randomized_from_corpus(conf, conf.corpus, 2)
    src_batch = trains[0]
    results, loss = runner.forward(src_batch, conf, encdec, True, 100)
    # print(results)
    assert loss.data > 0.0

def test_train_v1(test_conf_v1):
    conf = test_conf_v1
    encdec = EncoderDecoder(conf)
    runner.train(conf)
    assert True

# -------------------- v2 -----------------------------------------------------------
def test_forward_v2(test_conf_v2):
    conf = test_conf_v2
    encdec = EncoderDecoder(conf)
    train_idxs, test_idxs, trains, tests = encdec.minbatch_class.randomized_from_corpus(conf, conf.corpus, 2)
    src_batch = trains[0]
    results, loss = runner.forward(src_batch, conf, encdec, True, 100)
    # print(results)
    assert loss.data > 0.0

def test_train_v2(test_conf_v2):
    conf = test_conf_v2
    encdec = EncoderDecoder(conf)
    runner.train(conf)
    assert True
