from __future__ import unicode_literals
import numpy as np
import matplotlib.pyplot as plt

import chainer
from chainer import Chain, Variable, cuda, optimizer, optimizers, serializers
import chainer.functions as F
import chainer.links as L

import config
import encdec
from minbatch import MinBatch

def forward(batch, conf, encdec, is_training, generation_limit):
  batch_size = batch.batch_size()
  hyp_batch  = [[] for _ in range(batch_size)]
  encdec.reset(batch_size)

  for seq_idx in reversed(range(batch.data_seq_length())):
    x = batch.data_batch_at(seq_idx)
    encdec.encode(x)

  t = batch.boundary_symbol_batch()
  if is_training:
    for seq_idx in range(batch.teach_seq_length()):
      y = encdec.decode(t)
      t = batch.teach_batch_at(seq_idx)
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
  epoch_train_blue_scores = []
  epoch_test_blue_scores  = []
  corpus = conf.corpus

  for epoch in range(conf.epoch()):
    logging('epoch %d/%d: ' % (epoch+1, conf.epoch()))
    trained = 0
    train_blue_scores = []
    test_blue_scores  = []
    train_idxs, test_idxs, trains, tests = MinBatch.randomized_from_corpus(conf, conf.corpus, conf.batch_size())

    for batch in trains:
      hyp_batch, loss = forward(batch, conf, encdec, True, 0)
      loss.backward()
      opt.update()
      trained += batch.batch_size()
      scores = report_batch(conf, corpus, epoch, trained, batch, hyp_batch, '--- TRAIN -------')
      train_blue_scores += scores

    for batch in tests:
      hyp_batch = forward(batch, conf, encdec, False, 15)
      scores = report_batch(conf, corpus, epoch, trained, batch, hyp_batch, '--- TEST -------')
      test_blue_scores += scores

    report_epoch(conf, epoch, train_blue_scores, test_blue_scores)
    epoch_train_blue_scores.append( np.array(train_blue_scores).mean() )
    epoch_test_blue_scores.append(  np.array(test_blue_scores).mean() )
    if (epoch % 10) == 0:
      save(conf, encdec, epoch)

  return epoch_train_blue_scores, epoch_test_blue_scores

def predict(conf, encdec, source):
  batch = MinBatch.from_text(conf, conf.corpus, source)
  print ' '.join(conf.corpus.ids_to_tokens(batch.data_at(0)))
  hyp   = forward(batch, conf, encdec, False, 30)
  print ' '.join(hyp[0])
  # return hyp

def report_epoch(conf, epoch, train_blue_scores, test_blue_scores):
  train_mean = np.array(train_blue_scores).mean()
  test_mean  = np.array(test_blue_scores).mean()
  logging('================================================================')
  logging('finish epoch %3d/%3d - train BLEU: %.3f - test BLEU: %.3f' % (epoch + 1, conf.epoch(), train_mean, test_mean))
  logging('================================================================')
  logging('')

def report_batch(conf, corpus, epoch, trained, batch, hyp_batch, header):
  bleu_scores = []
  for k in range(batch.batch_size()):
    data_tokens  = corpus.ids_to_tokens(batch.data_at(k))
    teach_tokens = corpus.ids_to_tokens(batch.teach_at(k))
    hyp_tokens   = hyp_batch[k]
    bleu_score   = corpus.bleu_score(hyp_tokens, [teach_tokens])
    bleu_scores.append(bleu_score)
    logging(header)
    logging('epoch %3d/%3d, sample %8d' % (epoch + 1, conf.epoch(), trained))
    logging('  source  = ' + ' '.join(data_tokens))
    logging('  teacher = ' + ' '.join(teach_tokens))
    logging('  predict = ' + ' '.join(hyp_tokens))
    logging('  BLEU    = {0:.3f}'.format(bleu_score))
  return bleu_scores

def report_bleu_graph(train_blue_scores, test_blue_scores):
  plt.plot(train_blue_scores)
  plt.plot(test_blue_scores)
  plt.show()
  # plt.savefig("image.png")

def save(conf, encdec, epoch):
  conf.save('model/', encdec, epoch)

def load(prefix):
  encdec, opt, conf = config.Config.load(prefix)
  return encdec, opt, conf

def logging(log):
  print(log)
