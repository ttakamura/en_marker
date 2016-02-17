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

def pytest_funcarg__encdec_s(request, test_corp):
    args = "--mode train --embed 50 --hidden 30 --minbatch 15".split(" ")
    conf = config.parse_args(raw_args = args)
    conf.corpus = test_corp
    return EncoderDecoder(conf)

def pytest_funcarg__test_corp(request):
    return corpus.open(test_file)

def test_forward(encdec_s, test_corp):
    conf = encdec_s.conf
    conf.corpus = test_corp
    train_idxs, test_idxs, trains, tests = MinBatch.randomized_from_corpus(conf, test_corp, 2)
    src_batch = trains[0]
    trg_batch = tests[0]
    results, loss = runner.forward(src_batch, trg_batch, conf, encdec_s, True, 100)
    # print(results)
    assert loss.data > 0.0
