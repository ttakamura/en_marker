from __future__ import unicode_literals
import pytest
import sys
sys.path.append('lib')

import config
import corpus
import runner
from minbatch import MinBatch, MarkTeacherMinBatch
from encdec import EncoderDecoder, Encoder, MarkDecoder

test_file = "tests/test.html"

def pytest_funcarg__test_corp(request):
    return corpus.open(test_file)

def parse_conf(model, test_corp):
    args = "--mode train --embed 50 --hidden 30 --minbatch 15".split(" ")
    args += ["--model", model]
    conf = config.parse_args(raw_args = args)
    conf.corpus = test_corp
    return conf

def build_model(model, test_corp):
    conf = parse_conf(model, test_corp)
    encdec, opt = conf.setup_model()
    return conf, encdec, opt

def dummy_data_train(conf, encdec, opt, batch):
    prev_loss = 100.0
    for i in range(40):
        hyp, loss = encdec.forward(conf, batch, True)
        loss.backward()
        opt.update()
        loss = float(loss.data)
        print loss
        prev_loss = loss
    return prev_loss

# -------------------------------------------------------------
def test_v2_model(test_corp):
    conf, encdec, opt = build_model("v2", test_corp)
    enc = encdec.enc
    assert type(enc) == Encoder
    assert enc.xe.W.data.shape == (conf.vocab_size(), 50)
    assert enc.eh.W.data.shape == (120, 50)
    assert enc.hh.W.data.shape == (120, 30)
    dec = encdec.dec
    assert type(dec) == MarkDecoder
    assert dec.ye.W.data.shape == (conf.vocab_size(), 50)
    assert dec.eh.W.data.shape == (120, 50)
    assert dec.hh.W.data.shape == (120, 30)
    assert dec.hf.W.data.shape == (conf.mark_dim_size(),  30)

def test_v2_train(test_corp):
    conf, encdec, opt = build_model("v2", test_corp)
    batch_size = 2
    train_idxs, test_idxs, trains, tests = MarkTeacherMinBatch.randomized_from_corpus(conf, conf.corpus, batch_size)
    loss = dummy_data_train(conf, encdec, opt, trains[0])
    assert loss < 5.0
