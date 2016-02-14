from __future__ import unicode_literals
import numpy as np

import chainer
from chainer import Chain, Variable, cuda, optimizer, optimizers, serializers
import chainer.functions as F
import chainer.links as L

import config
import encdec

def forward(src_batch, trg_batch, corpus, conf, encdec, is_training, generation_limit):
  xp = conf.xp()
  batch_size = len(src_batch)
  src_len    = len(src_batch[0])
  trg_len    = len(trg_batch[0]) if trg_batch else 0
  encdec.reset(batch_size)

  for l in reversed(range(src_len)):
    x = xp.array([src_stoi(src_batch[k][l]) for k in range(batch_size)], dtype=np.int32)
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
