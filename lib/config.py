from __future__ import unicode_literals
import sys
import yaml
import numpy
from chainer import cuda, optimizers, optimizer, serializers

import corpus
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

    def __eq__(self, other):
        if other == None:
            return False
        return (self.model()       == other.model()) and \
               (self.train_file()  == other.train_file()) and \
               (self.embed_size()  == other.embed_size()) and \
               (self.hidden_size() == other.hidden_size()) and \
               (self.batch_size()  == other.batch_size()) and \
               (self.lr()          == other.lr())

    def __ne__(self, other):
        return not self == other

    def parse_args(self, raw_args):
        default_embed     = 100
        default_hidden    = 200
        default_epoch     = 10
        default_minbatch  = 64
        default_lr        = 0.01
        p = ArgumentParser(description='English marker')
        p.add_argument('--mode',       default='console',        help='console, train or test')
        p.add_argument('--model',      default='encdec',         help='encdec or hoge')
        p.add_argument('--gpu',        default=-1)
        p.add_argument('--embed',      default=default_embed,    type=int)
        p.add_argument('--hidden',     default=default_hidden,   type=int)
        p.add_argument('--epoch',      default=default_epoch,    type=int)
        p.add_argument('--minbatch',   default=default_minbatch, type=int)
        p.add_argument('--lr',         default=default_lr,       type=float)
        p.add_argument('--train_file', default='data/original.html')
        p.add_argument('--load_prefix', default='model/sample',  help='load from the model')
        p.add_argument('--minor_word', default=1,                type=int, help='minimum frequency of minor-word')
        return p.parse_args(raw_args)

    def save(self, prefix, encdec, epoch):
        settings = {
            "model":       self.model(),
            "train_file":  self.train_file(),
            "embed":       self.embed_size(),
            "hidden":      self.hidden_size(),
            "minbatch":    self.batch_size(),
            "lr":          self.lr(),
            "epoch":       epoch
        }
        prefix = prefix + "_".join([k+":"+str(v) for k,v in settings.items()]).replace("/", "-").replace(".", "-")
        self.corpus.save(prefix + '.vocab')
        self.serialize(prefix + '.conf', settings)
        serializers.save_npz(prefix + '.weights', encdec)
        return prefix

    @staticmethod
    def load(prefix, raw_args=None):
        conf = Config.deserialize(prefix + '.conf', raw_args=raw_args)
        conf.corpus = corpus.EnMarkCorpus.load(prefix + '.vocab')
        encdec, opt = conf.setup_model()
        serializers.load_npz(prefix + '.weights', encdec)
        return encdec, opt, conf

    def serialize(self, file_path, data):
        with open(file_path, 'w') as file:
            yaml.dump(data, file, encoding='utf8', allow_unicode=True)

    @staticmethod
    def deserialize(file_path, raw_args=None):
        data = None
        with open(file_path, 'r') as file:
            data = yaml.load(file.read())
        conf = parse_args(raw_args)
        conf.merge(data)
        return conf

    def merge(self, new_args):
        self.args.model      = new_args['model']
        self.args.train_file = new_args['train_file']
        self.args.embed      = new_args['embed']
        self.args.hidden     = new_args['hidden']
        self.args.minbatch   = new_args['minbatch']
        self.args.lr         = new_args['lr']

    def open_corpus(self):
        self.corpus = corpus.open(self.train_file(), tagger=corpus.perceptron_tagger())
        self.corpus.minor_word_frequency = self.minor_word()
        return self.corpus

    def load_prefix(self):
        return self.args.load_prefix

    def corpus(self):
        return self.corpus

    def xp(self):
        return cuda.cupy if self.use_gpu() else numpy

    def mode(self):
        return self.args.mode

    def model(self):
        return self.args.model

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

    def minor_word(self):
        return self.args.minor_word

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
