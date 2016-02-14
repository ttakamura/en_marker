from __future__ import unicode_literals
import pytest
import sys
sys.path.append('lib')

import config
from encdec import EncoderDecoder

def pytest_funcarg__encdec_s(request):
    args = "--mode train --embed 50 --hidden 30 --minbatch 15".split(" ")
    conf = config.parse_args(raw_args = args)
    return EncoderDecoder(100, conf)

def test_encdec(encdec_s):
    assert encdec_s.vocab_size  == 100

def test_encoder(encdec_s):
    enc = encdec_s.enc
    assert enc.xe.W.data.shape == (100, 50)

def test_decoder(encdec_s):
    dec = encdec_s.dec
    assert dec.ye.W.data.shape == (100, 50)
