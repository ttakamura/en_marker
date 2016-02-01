import sys
import argparse
import numpy as np
import code
sys.path.append('lib')

import corpus

parser = argparse.ArgumentParser(description='English marker')
parser.add_argument('--mode', default='console', help='console or test')
args = parser.parse_args()

if args.mode == 'console':
    code.interact(local=locals())
else:
    print 'hello'
