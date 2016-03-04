from __future__ import unicode_literals
import codecs
import numpy as np

class MinBatch:
    @staticmethod
    def randomized_from_corpus(conf, corpus, batch_size):
        train_size  = int(corpus.size() / 3 * 2) / batch_size
        test_size   = int(corpus.size() / 3 * 1) / batch_size
        test_offset = train_size * batch_size
        train_idxs  = np.random.permutation(train_size * batch_size).reshape(train_size, batch_size)
        test_idxs   = np.random.permutation(test_size  * batch_size).reshape(test_size,  batch_size) + test_offset
        trains      = MinBatch.from_corpus(conf, corpus, train_idxs)
        tests       = MinBatch.from_corpus(conf, corpus, test_idxs)
        return train_idxs, test_idxs, trains, tests

    @staticmethod
    def from_corpus(conf, corpus, idxs_list):
        batches = []
        for idxs in idxs_list:
            data_id_rows  = [corpus.data_at(i) for i in idxs]
            teach_id_rows = [corpus.teacher_at(i) for i in idxs]
            batch = MinBatch(conf, corpus, data_id_rows, teach_id_rows)
            batches.append(batch)
        return batches

    @staticmethod
    def from_text(conf, corpus, source):
        if not isinstance(source, list):
            # ["hello world"]
            source = [source]
        source = [corpus.encode(s) for s in source]
        return MinBatch(conf, corpus, source)

    def __init__(self, conf, corpus, data_id_rows, teach_id_rows=None):
        self.conf      = conf
        self.corpus    = corpus
        self.data_rows = self.fill_pad(data_id_rows)
        if teach_id_rows == None:
            self.teach_rows = None
        else:
            self.teach_rows = self.fill_pad(teach_id_rows)

    def __eq__(self, other):
        return (self.data_rows  == other.data_rows) and \
               (self.teach_rows == other.teach_rows) and \
               (self.corpus     == other.corpus)

    def __ne__(self, other):
        return not self == other

    def fill_pad(self, id_rows):
        pad_id     = self.corpus.token_to_id("<pad>")
        max_length = max([ len(row) for row in id_rows ])
        for row in id_rows:
            if max_length > len(row):
                pad_size = (max_length - len(row))
                for _ in range(pad_size):
                    row.append(pad_id)
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
        return self.teach_rows[idx]

    def teach_batch_at(self, seq_idx):
        xp = self.conf.xp()
        x  = xp.array([self.teach_rows[k][seq_idx] for k in range(self.batch_size())], dtype=np.int32)
        return x

    def batch_size(self):
        return len(self.data_rows)

    def data_seq_length(self):
        return len(self.data_rows[0])

    def teach_seq_length(self):
        return len(self.teach_rows[0])
