from __future__ import unicode_literals
import pytest
import sys
sys.path.append('lib')

import config
import corpus
from corpus import MinBatch
import runner
from encdec import EncoderDecoder

test_file = "tests/test.html"

def pytest_funcarg__test_conf(request):
    args = "--mode train --embed 50 --hidden 30 --minbatch 2".split(" ")
    conf = config.parse_args(raw_args = args)
    conf.corpus = corpus.open(test_file)
    return conf

def test_forward(test_conf):
    conf = test_conf
    encdec = EncoderDecoder(conf)
    train_idxs, test_idxs, trains, tests = MinBatch.randomized_from_corpus(conf, conf.corpus, 2)
    src_batch = trains[0]
    results, loss = runner.forward(src_batch, conf, encdec, True, 100)
    # print(results)
    assert loss.data > 0.0

def test_train(test_conf):
    conf = test_conf
    encdec = EncoderDecoder(conf)
    # runner.train(conf)
    assert True
