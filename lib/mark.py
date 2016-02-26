from __future__ import unicode_literals

def mark_dim_size():
    return 10

class WordMark:
    def __init__(self, raw_args):
        self.args = self.parse_args(raw_args)
        self.corpus = None
