# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import re
import numpy as np

mark_types = [
    '<->'   , # no anotation
    '<sj>'  , # subject
    '<ssj>' , # sub-subject
    '<v>'   , # verb
    '<not>' , # negative form
    '<av>'  , # auxiliary verb
    '<cp>'  , # continuous participle
    '<pp>'  , # past participle
    '<fp>'  , # future participle
    # '<s>'   , # sentence
    # '<ss>'  , # sub-sentence
    # '<rel>' ,  # relational
    '<con>'   # conjuections
]

open_type_to_idx_map  = {type: idx for idx, type in enumerate(mark_types)}
open_type_to_idx_map['<pad>'] = -1

close_type_to_idx_map = {re.sub(r'<(.+)>', r'</\1>', type): idx for idx, type in enumerate(mark_types)}
close_type_to_idx_map['</pad>'] = -1

def mark_dim_size():
    return len(mark_types)

def idx_to_type(idx):
    if idx == -1:
        return '....'
    elif mark_types[idx] == '<->':
        return '____'
    else:
        return mark_types[idx]

def convert_teach_id_row(row, corpus):
    context   = {type: False for type in mark_types}
    mark_vecs = []
    for id in row:
        token = corpus.id_to_token(id)
        if corpus.is_teacher_tag(id):
            if open_type_to_idx_map.has_key(token):
                context[token] = True
            elif close_type_to_idx_map.has_key(token):
                context[mark_types[close_type_to_idx_map[token]]] = False
        else:
            types = [type for type, flag in context.items() if flag]
            vec   = convert_types_to_vec(types)
            mark_vecs.append(vec)
    return mark_vecs

def convert_types_to_vec(type_tokens):
    vec = np.zeros(mark_dim_size(), dtype=np.float32)
    for type in type_tokens:
        vec[open_type_to_idx_map[type]] = 1.0
    if len(type_tokens) == 0:
        vec = open_type_to_idx_map['<->']   # n-hot => 1-hot
    else:
        vec = vec.argmax(0)                 # n-hot => 1-hot
    return vec

def padding():
    vec = open_type_to_idx_map['<pad>']     # n-hot => 1-hot
    return vec

def decoded_vec_score(t, y):
    y_max = argmax_vec(y)
    if t.ndim == 1:
        t_max = t
    else:
        t_max = t.argmax(1)
    score = 0.0
    total = 0.0
    for k in range(t_max.shape[0]):
        if t_max[k] != -1:
            if y_max[k] == t_max[k]:
                if open_type_to_idx_map['<->'] != t_max[k]:
                    total += 1.0
                    score += 1.0
            else:
                total += 1.0
    if total == 0.0:
        return 1.0
    else:
        return score / total

def decoded_vec_to_str(y):
    result = []
    if y.ndim == 1:
        output = y
        for k in range(len(output)):
            result.append(idx_to_type(output[k]))
    else:
        output = argmax_vec(y)
        for k in range(len(output)):
            result.append(idx_to_type(output[k]))
    return result

def decoded_vec_to_hash(y):
    result = []
    output = argmax_vec(y)
    for k in range(len(output)):
        # TODO: 出力を足切りして多次元にする
        mark = { mark_types[output[k]]: True }
        result.append(mark)
    return result

def argmax_vec(y):
    # probs  = y.max(1)
    output = y.argmax(1)
    # output[probs < 0.3] = -1
    return output
