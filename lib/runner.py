from __future__ import unicode_literals
import numpy as np

import chainer
from chainer import Chain, Variable, cuda, optimizer, optimizers, serializers
import chainer.functions as F
import chainer.links as L

import config
import encdec

def forward(src_batch, trg_batch, conf, encdec, is_training, generation_limit):
  xp = conf.xp()
  encdec.reset(src_batch.batch_size())

  for seq_idx in reversed(range(src_batch.seq_length())):
    x = src_batch.batch_at(seq_idx)
    encdec.encode(x)

  t = src_batch.boundary_symbol_batch()
  hyp_batch = [[] for _ in range(batch_size)]

  if is_training:
    loss = Variable(xp.zeros((), dtype=np.float32))
    for seq_idx in range(trg_batch.seq_length()):
      y = encdec.decode(t)
      t = trg_batch.batch_at(seq_idx)
      loss += F.softmax_cross_entropy(y, t)
      output = cuda.to_cpu(y.data.argmax(1))
      for k in range(trg_batch.batch_size()):
        hyp_batch[k].append( conf.corpus.id_to_token(output[k]) )
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
