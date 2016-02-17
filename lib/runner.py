from __future__ import unicode_literals
import numpy as np

import chainer
from chainer import Chain, Variable, cuda, optimizer, optimizers, serializers
import chainer.functions as F
import chainer.links as L

import config
import encdec
from corpus import MinBatch

def forward(src_batch, trg_batch, conf, encdec, is_training, generation_limit):
  xp         = conf.xp()
  batch_size = src_batch.batch_size()
  t          = src_batch.boundary_symbol_batch()
  hyp_batch  = [[] for _ in range(batch_size)]
  encdec.reset(batch_size)

  for seq_idx in reversed(range(src_batch.seq_length())):
    x = src_batch.batch_at(seq_idx)
    encdec.encode(x)

  if is_training:
    for seq_idx in range(trg_batch.seq_length()):
      y = encdec.decode(t)
      t = trg_batch.batch_at(seq_idx)
      encdec.add_loss(y, t)
      output = cuda.to_cpu(y.data.argmax(1))
      for k in range(batch_size):
        hyp_batch[k].append( conf.corpus.id_to_token(output[k]) )
    return hyp_batch, encdec.loss

  else:
    while len(hyp_batch[0]) < generation_limit:
      y = encdec.decode(t)
      output = cuda.to_cpu(y.data.argmax(1))
      t = np.array(output, dtype=np.int32)
      for k in range(batch_size):
        hyp_batch[k].append( conf.corpus.id_to_token(output[k]) )
      if all(hyp_batch[k][-1] == '<eos>' for k in range(batch_size)):
        break
    return hyp_batch

def train(conf):
  encdec, opt = conf.setup_model()

  for epoch in range(conf.epoch()):
    logging('epoch %d/%d: ' % (epoch+1, conf.epoch()))
    trained = 0
    train_idxs, test_idxs, trains, tests = MinBatch.randomized_from_corpus(conf, conf.corpus, conf.batch_size())

    for src_batch, trg_batch in range(conf.corpus.):
      src_batch = fill_batch(src_batch)
      trg_batch = fill_batch(trg_batch)
      K = len(src_batch)
      hyp_batch, loss = forward(src_batch, trg_batch, src_vocab, trg_vocab, encdec, True, 0)
      loss.backward()
      opt.update()

      for k in range(K):
        trace('epoch %3d/%3d, sample %8d' % (epoch + 1, conf.epoch(), trained + k + 1))
        trace('  src = ' + ' '.join([x if x != '</s>' else '*' for x in src_batch[k]]))
        trace('  trg = ' + ' '.join([x if x != '</s>' else '*' for x in trg_batch[k]]))
        trace('  hyp = ' + ' '.join([x if x != '</s>' else '*' for x in hyp_batch[k]]))

      trained += K

    trace('saving model ...')
    prefix = args.model + '.%03.d' % (epoch + 1)
    src_vocab.save(prefix + '.srcvocab')
    trg_vocab.save(prefix + '.trgvocab')
    encdec.save_spec(prefix + '.spec')
    serializers.save_hdf5(prefix + '.weights', encdec)

def logging(log):
  print(log)
