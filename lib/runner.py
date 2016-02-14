from __future__ import unicode_literals
import numpy as np

import chainer
from chainer import Chain, Variable, cuda, optimizer, optimizers, serializers
import chainer.functions as F
import chainer.links as L

import config
import encdec

class MinBatch:
    def __init__(self, conf, id_rows):
        self.conf = conf
        self.rows = self.fill_pad(id_rows)

    def fill_pad(self, id_rows):
        return id_rows

    def batch_at(self, seq_idx):
        xp = self.conf.xp()
        x  = xp.array([self.id_rows[k][l] for k in range(batch_size)], dtype=np.int32)
        return x

    def batch_size(self):
        return len(self.rows)

    def seq_length(self):
        return len(self.rows[0])

def forward(src_batch, trg_batch, conf, encdec, is_training, generation_limit):
  xp = conf.xp()
  encdec.reset(src_batch.batch_size())

  for l in reversed(range(src_len)):
    x = src_batch.batch_at(l)
    encdec.encode(x)

  t = XP.iarray([trg_stoi('<s>') for _ in range(batch_size)])
  hyp_batch = [[] for _ in range(batch_size)]

  if is_training:
    loss = XP.fzeros(())
    for l in range(trg_len):
      y = encdec.decode(t)
      t = XP.iarray([trg_stoi(trg_batch[k][l]) for k in range(batch_size)])
      loss += functions.softmax_cross_entropy(y, t)
      output = cuda.to_cpu(y.data.argmax(1))
      for k in range(batch_size):
        hyp_batch[k].append(trg_itos(output[k]))
    return hyp_batch, loss

  else:
    while len(hyp_batch[0]) < generation_limit:
      y = encdec.decode(t)
      output = cuda.to_cpu(y.data.argmax(1))
      t = XP.iarray(output)
      for k in range(batch_size):
        hyp_batch[k].append(trg_itos(output[k]))
      if all(hyp_batch[k][-1] == '</s>' for k in range(batch_size)):
        break

    return hyp_batch
