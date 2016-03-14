from __future__ import unicode_literals
import codecs
import numpy as np

import mark

class MinBatch:
    @classmethod
    def randomized_from_corpus(cls, conf, corpus, batch_size):
        train_size  = int(corpus.size() / 3 * 2) / batch_size
        test_size   = int(corpus.size() / 3 * 1) / batch_size
        test_offset = train_size * batch_size
        train_idxs  = np.random.permutation(train_size * batch_size).reshape(train_size, batch_size)
        test_idxs   = np.random.permutation(test_size  * batch_size).reshape(test_size,  batch_size) + test_offset
        trains      = cls.from_corpus(conf, corpus, train_idxs)
        tests       = cls.from_corpus(conf, corpus, test_idxs)
        return train_idxs, test_idxs, trains, tests

    @classmethod
    def from_corpus(cls, conf, corpus, idxs_list):
        batches = []
        for idxs in idxs_list:
            data_id_rows  = [corpus.data_at(i) for i in idxs]
            teach_id_rows = [corpus.teacher_at(i) for i in idxs]
            # for k in range(len(data_id_rows)):
            #     print corpus.ids_to_tokens(data_id_rows[k])
            #     print corpus.ids_to_tokens(teach_id_rows[k])
            batch = cls(conf, corpus, data_id_rows, teach_id_rows)
            batches.append(batch)
        return batches

    @classmethod
    def from_text(cls, conf, corpus, source):
        if not isinstance(source, list):
            source = [source]
        source = [corpus.encode(s) for s in source]
        return cls(conf, corpus, source)

    def __init__(self, conf, corpus, data_id_rows, teach_id_rows=None):
        self.conf      = conf
        self.corpus    = corpus
        self.data_rows = self.fill_pad(data_id_rows, self.corpus.token_to_id("<pad>"))
        if teach_id_rows == None:
            self.teach_rows = None
        else:
            self.teach_rows = self.convert_teach_id_rows(teach_id_rows)
        self.teach_dtype = np.int32

    def __eq__(self, other):
        return (self.data_rows  == other.data_rows) and \
               (self.teach_rows == other.teach_rows) and \
               (self.corpus     == other.corpus)

    def __ne__(self, other):
        return not self == other

    def convert_teach_id_rows(self, id_rows):
        return self.fill_pad(id_rows, self.corpus.token_to_id("<pad>"))

    def fill_pad(self, id_rows, padding):
        max_length = max([ len(row) for row in id_rows ])
        for row in id_rows:
            if max_length > len(row):
                pad_size = (max_length - len(row))
                for _ in range(pad_size):
                    row.append(padding)
        return id_rows

    def boundary_symbol_batch(self):
        # Do I need return a special charactor?
        return self.data_batch_at(0)

    def data_at(self, idx):
        return self.data_rows[idx]

    def data_batch_at(self, seq_idx):
        xp = self.conf.xp()
        x  = xp.array([self.data_rows[k][seq_idx] for k in range(self.batch_size())], dtype=np.int32)
        return x

    def teach_at(self, idx):
        if self.teach_rows == None:
            return None
        else:
            return self.teach_rows[idx]

    def teach_batch_at(self, seq_idx):
        xp = self.conf.xp()
        x  = xp.array([self.teach_rows[k][seq_idx] for k in range(self.batch_size())], dtype=self.teach_dtype)
        return x

    def batch_size(self):
        return len(self.data_rows)

    def data_seq_length(self):
        return len(self.data_rows[0])

    def teach_seq_length(self):
        return len(self.teach_rows[0])

# teacher = [ mark-vector ... ]
class MarkTeacherMinBatch(MinBatch):
    def __init__(self, conf, corpus, data_id_rows, teach_id_rows=None):
        MinBatch.__init__(self, conf, corpus, data_id_rows)
        if teach_id_rows == None:
            self.teach_rows = None
        else:
            self.teach_rows = self.convert_teach_id_rows(teach_id_rows)
        self.teach_dtype = np.float32

    def convert_teach_id_rows(self, id_rows):
        rows = [mark.convert_teach_id_row(row, self.corpus) for row in id_rows]
        return self.fill_pad(rows, mark.padding())
