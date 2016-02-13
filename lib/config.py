import sys
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

def mode():
    return parse_args().args.mode

def gpu():
    return parse_args().args.gpu

def use_gpu():
    return parse_args().args.gpu > 0

def embed():
    return parse_args().args.embed

def hidden():
    return parse_args().args.hidden

def epoch():
    return parse_args().args.epoch

def minbatch():
    return parse_args().args.minbatch

class Config:
    def __init__(self, raw_args):
        self.args = self.parse_args(raw_args)

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
