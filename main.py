import sys
import argparse
import numpy as np
from IPython import embed

sys.path.append('lib')
import corpus
import config
import runner

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
else:
    print 'hello'
