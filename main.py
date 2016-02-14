import sys
import argparse
import numpy as np
from IPython import embed

sys.path.append('lib')
import corpus
import config

conf = config.parse_args()
conf.corpus = "hoge"

if conf.mode() == 'console':
    embed()
else:
    print 'hello'
