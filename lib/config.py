from __future__ import unicode_literals
import sys
import numpy
from chainer import cuda, optimizers, optimizer

from argparse import ArgumentParser
from encdec import EncoderDecoder

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
        default_lr        = 0.01
        p = ArgumentParser(description='English marker')
        p.add_argument('--mode',     default='console',        help='console, train or test')
        p.add_argument('--gpu',      default=-1)
        p.add_argument('--embed',    default=default_embed,    type=int)
        p.add_argument('--hidden',   default=default_hidden,   type=int)
        p.add_argument('--epoch',    default=default_epoch,    type=int)
        p.add_argument('--minbatch', default=default_minbatch, type=int)
        p.add_argument('--lr',       default=default_lr,       type=float)
        p.add_argument('--train_file')
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

    def vocab_size(self):
        return self.corpus.vocab_size()

    def lr(self):
        return self.args.lr

    def train_file(self):
        return self.args.train_file

    def encdec(self):
        return EncoderDecoder(self)

    def optimizer(self):
        return optimizers.AdaGrad(lr = self.lr())

    def setup_model(self):
        encdec = self.encdec()
        opt    = self.optimizer()
        opt.setup(encdec)
        opt.add_hook(optimizer.GradientClipping(5))
        if self.use_gpu():
            encdec.to_gpu()
        return encdec, opt
