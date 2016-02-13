import sys
import argparse
import numpy as np
from IPython import embed

sys.path.append('lib')
import corpus
import config

if config.mode() == 'console':
    embed()
else:
    print 'hello'
