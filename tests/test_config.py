from __future__ import unicode_literals
import pytest
import sys
sys.path.append('lib')

import config

def pytest_funcarg__config_a(request):
    return config.parse_args(raw_args = ['--mode', 'train'])

def test_config_gpu(config_a):
    assert config_a.gpu() == -1
    assert config_a.use_gpu() == False
