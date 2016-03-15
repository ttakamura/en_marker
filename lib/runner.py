from __future__ import unicode_literals
import numpy as np
import matplotlib.pyplot as plt

import chainer
from chainer import Chain, Variable, cuda, optimizer, optimizers, serializers
import chainer.functions as F
import chainer.links as L

import config
import encdec
from minbatch import MarkTeacherMinBatch

def forward(batch, conf, encdec, is_training, generation_limit):
  hyp_batch, loss = encdec.forward(conf, batch, is_training, generation_limit)
  if is_training:
    return hyp_batch, encdec.loss
  else:
    return hyp_batch

def train(conf):
  encdec, opt = conf.setup_model()
  epoch_train_scores = []
  epoch_test_scores  = []
  corpus = conf.corpus

  for epoch in range(conf.epoch()):
    logging('epoch %d/%d: ' % (epoch+1, conf.epoch()))
    trained = 0
    train_scores = []
    test_scores  = []
    train_idxs, test_idxs, trains, tests = encdec.minbatch_class.randomized_from_corpus(conf, conf.corpus, conf.batch_size())

    for batch in trains:
      hyp_batch, loss = forward(batch, conf, encdec, True, 0)
      loss.backward()
      opt.update()
      trained += batch.batch_size()
      scores = report_batch(conf, corpus, epoch, trained, batch, hyp_batch, '--- TRAIN -------')
      train_scores += scores

    for batch in tests:
      hyp_batch = forward(batch, conf, encdec, False, 15)
      scores = report_batch(conf, corpus, epoch, trained, batch, hyp_batch, '--- TEST -------')
      test_scores += scores

    report_epoch(conf, epoch, train_scores, test_scores)
    epoch_train_scores.append( np.array(train_scores).mean() )
    epoch_test_scores.append(  np.array(test_scores).mean() )
    if (epoch % 10) == 0:
      save(conf, encdec, epoch)

  return epoch_train_scores, epoch_test_scores

def predict(conf, encdec, source):
  batch = encdec.minbatch_class.from_text(conf, conf.corpus, source)
  print ' '.join(conf.corpus.ids_to_tokens(batch.data_at(0)))
  hyp   = forward(batch, conf, encdec, False, 30)
  print ' '.join(hyp[0])
  # return hyp

def report_epoch(conf, epoch, train_blue_scores, test_blue_scores):
  train_mean = np.array(train_blue_scores).mean()
  test_mean  = np.array(test_blue_scores).mean()
  logging('================================================================')
  logging('finish epoch %3d/%3d - train score: %.3f - test score: %.3f' % (epoch + 1, conf.epoch(), train_mean, test_mean))
  logging('================================================================')
  logging('')

def report_batch(conf, corpus, epoch, trained, batch, hyp_batch, header):
  scores = []
  for k in range(batch.batch_size()):
    data_tokens = corpus.ids_to_tokens(batch.data_at(k))
    t, y = hyp_batch[k]
    if t[0] != None:
      masked_t = np.copy(t)
      masked_t
      score = np.sum(y * masked_t) / np.sum(masked_t)
      scores.append(score)
    logging(header)
    logging('epoch %3d/%3d, sample %8d' % (epoch + 1, conf.epoch(), trained))
    logging('  source  = ' + ' '.join(data_tokens))
    # logging('  teacher = ' + ' '.join())
    logging('  predict = ' + ' '.join(hyp_tokens))
  return scores

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
