from __future__ import unicode_literals
import pytest
import sys
sys.path.append('lib')

import config
import corpus
from encdec import EncoderDecoder

test_file = "tests/test.html"

def pytest_funcarg__test_corp(request):
    return corpus.open(test_file)

def pytest_funcarg__config_a(request, test_corp):
    conf = config.parse_args(raw_args = [
        '--mode',       'train',
        '--model',      'test_encdec',
        '--embed',      '128',
        '--hidden',     '64',
        '--minbatch',   '33',
        '--lr',         '0.3',
        '--train_file', 'data/test.html'
    ])
    conf.corpus = test_corp
    return conf

def test_config_gpu(config_a):
    assert config_a.gpu() == -1
    assert config_a.use_gpu() == False

def test_config_save_and_load(config_a):
    encdec = EncoderDecoder(config_a)
    prefix = config_a.save('tmp/test_conf', encdec, 10)
    print prefix
    new_conf = config.Config.load(prefix, raw_args=['--mode', 'test'])
    assert config_a == new_conf
