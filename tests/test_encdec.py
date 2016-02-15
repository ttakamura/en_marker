from __future__ import unicode_literals
import pytest
import sys
sys.path.append('lib')

import config
import corpus
import runner
from encdec import EncoderDecoder

test_file = "tests/test.html"

def pytest_funcarg__encdec_s(request):
    args = "--mode train --embed 50 --hidden 30 --minbatch 15".split(" ")
    conf = config.parse_args(raw_args = args)
    return EncoderDecoder(100, conf)

def pytest_funcarg__test_corp(request):
    return corpus.open(test_file)
