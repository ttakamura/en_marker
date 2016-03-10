from __future__ import unicode_literals
import numpy as np

import chainer
from chainer import Chain, Variable, cuda, optimizer, optimizers, serializers
import chainer.functions as F
import chainer.links as L

import config
import mark
from minbatch import MinBatch, MarkTeacherMinBatch

# -- Encoder -----------------------------------------------------------------------
class Encoder(Chain):
  @staticmethod
  def build(conf):
    return Encoder(conf.vocab_size(), conf.embed_size(), conf.hidden_size())

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

# -- Decoder -----------------------------------------------------------------------
# var-length decoder, output is a word-id (string)
class WordDecoder(Chain):
  @staticmethod
  def build(conf):
    return WordDecoder(conf.vocab_size(), conf.embed_size(), conf.hidden_size())

  def __init__(self, vocab_size, embed_size, hidden_size):
    super(WordDecoder, self).__init__(
        ye = L.EmbedID(vocab_size, embed_size),
        eh = L.Linear(embed_size, 4 * hidden_size),
        hh = L.Linear(hidden_size, 4 * hidden_size),
        hf = L.Linear(hidden_size, embed_size),
        fy = L.Linear(embed_size, vocab_size),
    )

  def __call__(self, y, c, h):
    e      = F.tanh(self.ye(y))
    c2, h2 = F.lstm(c, self.eh(e) + self.hh(h))
    f      = F.tanh(self.hf(h2))
    y2     = self.fy(f)
    return y2, c2, h2

  def train_input_batch_at(self, seq_idx, batch):
    if seq_idx == 0:
      return batch.boundary_symbol_batch()
    else:
      return batch.teach_batch_at(seq_idx)

  def predict_input_batch_at(self, seq_idx, y, batch):
    if seq_idx == 0:
      return batch.boundary_symbol_batch()
    else:
      output = cuda.to_cpu(y.data.argmax(1))
      t = np.array(output, dtype=np.int32)
      return t

  def decoded_vec_to_str(self, y, conf, batch_size):
    result = []
    output = cuda.to_cpu(y.data.argmax(1))
    for k in range(batch_size):
      result.append(conf.corpus.id_to_token(output[k]))
    return result

# fix-length decoder, output is a marking vector [0, 1, 0, 0, 0]
class MarkDecoder(Chain):
  @staticmethod
  def build(conf):
    return MarkDecoder(conf.vocab_size(), conf.embed_size(), conf.hidden_size(), conf.mark_dim_size())

  def __init__(self, vocab_size, embed_size, hidden_size, mark_size):
    super(MarkDecoder, self).__init__(
        ye = L.EmbedID(vocab_size, embed_size),
        eh = L.Linear(embed_size, 4 * hidden_size),
        hh = L.Linear(hidden_size, 4 * hidden_size),
        hf = L.Linear(hidden_size, mark_size)
    )

  def __call__(self, y, c, h):
    e      = F.tanh(self.ye(y))
    c2, h2 = F.lstm(c, self.eh(e) + self.hh(h))
    y2     = F.tanh(self.hf(h2))
    return y2, c2, h2

  def train_input_batch_at(self, seq_idx, batch):
    return batch.data_batch_at(seq_idx)

  def predict_input_batch_at(self, seq_idx, y, batch):
    return batch.data_batch_at(seq_idx)

  def decoded_vec_to_str(self, y, conf, batch_size):
    result = []
    output = cuda.to_cpu(y.data.argmax(1))
    for k in range(batch_size):
      result.append(mark.idx_to_type(k))
    return result

# ----------------------------------------------------------------------------
class EncoderDecoder(Chain):
  @staticmethod
  def build(conf):
    enc = Encoder
    if conf.model() == 'v1':
      dec   = WordDecoder
      batch = MinBatch
    elif  conf.model() == 'v2':
      dec   = MarkDecoder
      batch = MarkTeacherMinBatch
    else:
      print "Default model is used"
      dec   = WordDecoder # default
      batch = MinBatch
    return EncoderDecoder(conf, enclass=enc, declass=dec, minbatch_class=batch)

  def __init__(self, conf, enclass=Encoder, declass=WordDecoder, minbatch_class=MinBatch):
    super(EncoderDecoder, self).__init__(
        enc = enclass.build(conf),
        dec = declass.build(conf),
    )
    self.vocab_size = conf.vocab_size()
    self.conf = conf
    self.minbatch_class = minbatch_class

  def reset(self, batch_size):
    xp = self.conf.xp()
    self.zerograds()
    self.c    = Variable(xp.zeros((batch_size, self.conf.hidden_size()), dtype=np.float32))
    self.h    = Variable(xp.zeros((batch_size, self.conf.hidden_size()), dtype=np.float32))
    self.loss = Variable(xp.zeros((), dtype=np.float32))

  def encode(self, x):
    if type(x) != Variable:
      x = Variable(x)
    self.c, self.h = self.enc(x, self.c, self.h)

  def decode(self, y):
    if type(y) != Variable:
      y = Variable(y)
    y2, self.c, self.h = self.dec(y, self.c, self.h)
    return y2

  def add_loss(self, y, t):
    if type(t) != Variable:
      t = Variable(t)

    #print self.minbatch_class
    #print y.data
    #print t.data

    if t.data.ndim == 1:
      self.loss += F.softmax_cross_entropy(y, t)
    else:
      self.loss += -F.sum(F.log(F.softmax(y)) * t)

  def encode_seq(self, batch):
    for seq_idx in reversed(range(batch.data_seq_length())):
      x = batch.data_batch_at(seq_idx)
      self.encode(x)

  def decode_seq_train(self, conf, batch):
    result = []
    for seq_idx in range(batch.teach_seq_length()):
      x     = self.dec.train_input_batch_at(seq_idx, batch)
      y     = self.decode(x)
      t     = batch.teach_batch_at(seq_idx)
      y_str = self.dec.decoded_vec_to_str(y, conf, batch.batch_size())
      result.append((t, y, y_str))
    return result

  def decode_seq_predict(self, conf, batch, generation_limit):
    result = []
    y      = None
    while len(result) < generation_limit:
      x     = self.dec.predict_input_batch_at(seq_idx, y, batch)
      y     = self.decode(x)
      y_str = self.dec.decoded_vec_to_str(y, conf, batch.batch_size())
      result.append((y, y_str))
      if all(y_str[k] == '<eos>' for k in range(batch_size)):
        break
    return result

  def forward(self, conf, batch, is_training, generation_limit=100):
    batch_size = batch.batch_size()
    ys     = [[] for _ in range(batch_size)]
    ts     = [[] for _ in range(batch_size)]
    y_strs = [[] for _ in range(batch_size)]

    self.reset(batch_size)
    self.encode_seq(batch)

    if is_training:
      for t, y, y_str in self.decode_seq_train(conf, batch):
        self.add_loss(y, t)
        print t.data
        for k in range(batch_size):
          ys[k].append( y.data[k] )
          ts[k].append( t.data[k] )
          y_strs[k].append( y_str[k] )
    else:
      for y, y_str in self.decode_seq_predict(conf, batch, generation_limit):
        for k in range(batch_size):
          ys[k].append( y.data[k] )
          ts[k].append( None )
          y_strs[k].append( y_str[k] )

    hyp_batch = [(ts, ys, y_strs) for _ in range(batch_size)]
    return hyp_batch, self.loss
