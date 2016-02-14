from __future__ import unicode_literals
import sys
import numpy
from chainer import cuda

from argparse import ArgumentParser

main_conf = None

def parse_args(raw_args = None):
    global main_conf
    if raw_args == None:
        if main_conf == None:
            main_conf = Config(sys.argv[1:])
        return main_conf
    else:
        return Config(raw_args)

class Config:
    def __init__(self, raw_args):
        self.args = self.parse_args(raw_args)
        self.corpus = None

    def parse_args(self, raw_args):
        default_embed     = 100
        default_hidden    = 200
        default_epoch     = 10
        default_minbatch  = 64
        p = ArgumentParser(description='English marker')
        p.add_argument('--mode',     default='console',        help='console, train or test')
        p.add_argument('--gpu',      default=-1)
        p.add_argument('--embed',    default=default_embed,    type=int)
        p.add_argument('--hidden',   default=default_hidden,   type=int)
        p.add_argument('--epoch',    default=default_epoch,    type=int)
        p.add_argument('--minbatch', default=default_minbatch, type=int)
        return p.parse_args(raw_args)

    def corpus(self):
        return self.corpus

    def xp(self):
        return cuda.cupy if self.use_gpu() else numpy

    def mode(self):
        return self.args.mode

    def gpu(self):
        return self.args.gpu

    def use_gpu(self):
        return self.args.gpu > 0

    def embed_size(self):
        return self.args.embed

    def hidden_size(self):
        return self.args.hidden

    def epoch(self):
        return self.args.epoch

    def batch_size(self):
        return self.args.minbatch
