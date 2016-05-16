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
    # usage:
    # source = "this is a pen."
    # hyp    = runner.predict(conf, encdec, source)
    # t, y   = hyp[0]
    # mark.decoded_vec_to_str(y)
else:
    print 'hello'
