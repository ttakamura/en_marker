import sys
import argparse
import numpy as np
from IPython import embed

sys.path.append('lib')
import corpus
import config
import runner
import mark

conf   = config.parse_args()
corpus = conf.open_corpus()

if conf.mode() == 'console':
    embed()
elif conf.mode() == 'train':
    train_scores, test_scores = runner.train(conf)
    runner.report_bleu_graph(train_scores, test_scores)
elif conf.mode() == 'restore_console':
    encdec, opt, conf = runner.load(conf.load_prefix())
    embed()
    # usage: ---------------------------------------------------------------
    # source = "this is a pen."
    # batch, hyp = runner.predict(conf, encdec, source)
    # x = batch.data_at(0)
    # t, y = hyp[0]
    # mark.decoded_vec_to_str(y)
    #
    # In [24]: corpus.tokenize(source, cleanup_tag=False)
    # Out[24]: [u'<bos>', u'this', u'is', u'a', u'pen', u'.', u'<eos>']
    #
    # In [20]: corpus.ids_to_tokens(x)
    # Out[20]: [u'<bos>', u'this', u'is', u'a', u'<unk>', u'.', u'<eos>']
    #
    # In [21]: mark.decoded_vec_to_str(y)
    # Out[21]: [u'____', u'<sj>', u'<v>', u'____', u'____', u'____', u'____']

else:
    print 'hello'
