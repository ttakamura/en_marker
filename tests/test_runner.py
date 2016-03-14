from __future__ import unicode_literals
import pytest
import sys
sys.path.append('lib')

import config
import corpus
from minbatch import MinBatch, MarkTeacherMinBatch
import runner
from encdec import EncoderDecoder, Encoder, MarkDecoder

test_file = "tests/test.html"

def pytest_funcarg__test_conf_v2(request):
    args = "--mode train --embed 50 --hidden 30 --minbatch 2 --model v2 --epoch 2 --minor_word 0".split(" ")
    conf = config.parse_args(raw_args = args)
    conf.corpus = corpus.open(test_file)
    return conf

# -------------------- v2 -----------------------------------------------------------
def test_setup_v2(test_conf_v2):
    conf = test_conf_v2
    encdec, opt = conf.setup_model()
    assert type(encdec.enc)      == Encoder
    assert type(encdec.dec)      == MarkDecoder
    assert encdec.minbatch_class == MarkTeacherMinBatch

def test_forward_v2(test_conf_v2):
    conf = test_conf_v2
    encdec, opt = conf.setup_model()
    train_idxs, test_idxs, trains, tests = encdec.minbatch_class.randomized_from_corpus(conf, conf.corpus, 2)
    src_batch = trains[0]
    for _ in range(20):
        results, loss = runner.forward(src_batch, conf, encdec, True, 100)
        loss.backward()
        opt.update()
        print loss.data
    assert loss.data < 100.0

def test_train_v2(test_conf_v2):
    conf = test_conf_v2
    runner.train(conf)
    assert True
