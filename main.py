import sys
import argparse
import numpy as np
from IPython import embed

sys.path.append('lib')
import corpus
import config
import runner

conf = config.parse_args()
conf.corpus = corpus.open(conf.train_file())

if conf.mode() == 'console':
    embed()
elif conf.mode() == 'train':
    runner.train(conf)
else:
    print 'hello'
