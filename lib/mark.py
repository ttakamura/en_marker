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

def convert_teach_id_row(row):
    return []

def padding():
    return [[0]]
