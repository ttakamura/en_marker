from __future__ import unicode_literals

mark_map = [
    '-',
    'SJ',
    'V'
]

def mark_dim_size():
    return 10

def idx_to_type(idx):
    return mark_map[idx]

class WordMark:
    def __init__(self, raw_args):
        self.args = self.parse_args(raw_args)
        self.corpus = None
