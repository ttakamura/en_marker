from __future__ import unicode_literals
import numpy as np

import chainer
from chainer import Chain, Variable, cuda, optimizer, optimizers, serializers
import chainer.functions as F
import chainer.links as L

import config

class Encoder(Chain):
  def __init__(self, vocab_size, embed_size, hidden_size):
    super(Encoder, self).__init__(
        xe = L.EmbedID(vocab_size, embed_size),
        eh = L.Linear(embed_size, 4 * hidden_size),
        hh = L.Linear(hidden_size, 4 * hidden_size),
    )

  def __call__(self, x, c, h):
    e      = F.tanh(self.xe(x))
    c2, h2 = F.lstm(c, self.eh(e) + self.hh(h))
    return c2, h2

class Decoder(Chain):
  def __init__(self, vocab_size, embed_size, hidden_size):
    super(Decoder, self).__init__(
        ye = L.EmbedID(vocab_size, embed_size),
        eh = L.Linear(embed_size, 4 * hidden_size),
        hh = L.Linear(hidden_size, 4 * hidden_size),
        hf = L.Linear(hidden_size, embed_size),
        fy = L.Linear(embed_size, vocab_size),
    )

  def __call__(self, y, c, h):
    e      = F.tanh(self.ye(y))
    c2, h2 = F.lstm(c, self.eh(e) + self.hh(h))
    f      = F.tanh(self.hf(h))
    y2     = self.fy(f)
    return y2, c2, h2

class EncoderDecoder(Chain):
  def __init__(self, vocab_size, conf):
    super(EncoderDecoder, self).__init__(
        enc = Encoder(vocab_size, conf.embed_size(), conf.hidden_size()),
        dec = Decoder(vocab_size, conf.embed_size(), conf.hidden_size()),
    )
    self.vocab_size = vocab_size
    self.conf = conf

  def reset(self):
    xp = self.conf.xp()
    self.zerograds()
    self.c = xp.zeros((self.conf.batch_size(), self.conf.hidden_size()))
    self.h = xp.zeros((self.conf.batch_size(), self.conf.hidden_size()))

  def encode(self, x):
    self.c, self.h = self.enc(x, self.c, self.h)

  def decode(self, y):
    y2, self.c, self.h = self.dec(y, self.c, self.h)
    return y2
