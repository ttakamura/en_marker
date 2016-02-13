from __future__ import unicode_literals
import numpy as np

import chainer
from chainer import Chain, Variable, cuda, optimizer, optimizers, serializers
import chainer.functions as F
import chainer.links as L

import config
xp = cuda.cupy if config.use_gpu() else np

class Encoder(Chain):
  def __init__(self, vocab_size, embed_size, hidden_size):
    super(Encoder, self).__init__(
        xe = links.EmbedID(vocab_size, embed_size),
        eh = links.Linear(embed_size, 4 * hidden_size),
        hh = links.Linear(hidden_size, 4 * hidden_size),
    )

  def __call__(self, x, c, h):
    e = functions.tanh(self.xe(x))
    return functions.lstm(c, self.eh(e) + self.hh(h))

class Decoder(Chain):
  def __init__(self, vocab_size, embed_size, hidden_size):
    super(Decoder, self).__init__(
        ye = links.EmbedID(vocab_size, embed_size),
        eh = links.Linear(embed_size, 4 * hidden_size),
        hh = links.Linear(hidden_size, 4 * hidden_size),
        hf = links.Linear(hidden_size, embed_size),
        fy = links.Linear(embed_size, vocab_size),
    )

  def __call__(self, y, c, h):
    e = functions.tanh(self.ye(y))
    c, h = functions.lstm(c, self.eh(e) + self.hh(h))
    f = functions.tanh(self.hf(h))
    return self.fy(f), c, h

class EncoderDecoder(Chain):
  def __init__(self, vocab_size, embed_size, hidden_size):
    super(EncoderDecoder, self).__init__(
        enc = Encoder(vocab_size, embed_size, hidden_size),
        dec = Decoder(vocab_size, embed_size, hidden_size),
    )
    self.vocab_size = vocab_size
    self.embed_size = embed_size
    self.hidden_size = hidden_size

  def reset(self, batch_size):
    self.zerograds()
    self.c = XP.zeros((batch_size, self.hidden_size))
    self.h = XP.zeros((batch_size, self.hidden_size))

  def encode(self, x):
    self.c, self.h = self.enc(x, self.c, self.h)

  def decode(self, y):
    y, self.c, self.h = self.dec(y, self.c, self.h)
    return y

  def save_spec(self, filename):
    with open(filename, 'w') as fp:
      print(self.vocab_size, file=fp)
      print(self.embed_size, file=fp)
      print(self.hidden_size, file=fp)

  @staticmethod
  def load_spec(filename):
    with open(filename) as fp:
      vocab_size = int(next(fp))
      embed_size = int(next(fp))
      hidden_size = int(next(fp))
      return EncoderDecoder(vocab_size, embed_size, hidden_size)
