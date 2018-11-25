'''
用于数据库的DataBatch，next_batch方法与DataBatch兼容。
此类中的数据不保存在内存中，而是当next_batch方法调用时在数据库中查找
'''
import math

import numpy as np
import logging


class DataBatch_db(object):
    def __init__(self, db, input_vocabulary, output_vocabulary, id_list=None, shuffle=False,):
        self.db = db
        self._cursor = db.cursor()
        self.input_vocabulary = input_vocabulary
        self.output_vocabulary = output_vocabulary
        if id_list is not None:
            self._size = len(id_list)
            self._ids = id_list
        else:
            self._size = self._cursor.execute('SELECT count (*) AS num FROM conversation').fetchall()[0][0]
            self._ids = range(1, self._size + 1)

        self._ids = np.asarray(self._ids)
        if shuffle:
            index = np.random.permutation(self._size)
            self._ids = self._ids[index]
        self._shuffle = shuffle
        self._index = 0

    def batches(self, batch_size):
        index = self._index
        size = self._size
        for index in range(0, int(self._size / batch_size)):
            # if index + batch_size <= size:
            query = tuple(self._ids[index: index + batch_size])
            #     self._index += batch_size
            # else:
            #     query = tuple(np.concatenate((self._ids[index:], self._ids[0: index + batch_size - size])))
            #     self._index = index + batch_size - size
            #     if self._shuffle:
            #         index = np.random.permutation(self._size)
            #         self._ids = self._ids[index]

            value = self._cursor.execute('SELECT * from conversation where rowid in %s' % str(query)).fetchall()
            value = np.asarray(value)
            if len(value) != batch_size:
                logging.warning(
                    'number of result is less than expect while query the database with key={}'.format(str(query)))
            asks = value[:, 0].tolist()
            answers = value[:, 1].tolist()
            emotions = None # value[:, 2].tolist()
            asks = [sentence + " {0} ".format(self.input_vocabulary.EOS) for sentence in asks]
            answers = [sentence + " {0} ".format(self.input_vocabulary.EOS) for sentence in answers]
            asks = [self.input_vocabulary.words2ints(sentence) for sentence in asks]
            answers = [self.input_vocabulary.words2ints(sentence) for sentence in answers]
            seqlen_questions_in_batch = np.array([len(q) for q in asks])
            seqlen_answers_in_batch = np.array([len(a) for a in answers])

            padded_questions_in_batch = np.array(self._apply_padding(asks, self.input_vocabulary))
            padded_answers_in_batch = np.array(self._apply_padding(answers, self.output_vocabulary))

            yield padded_questions_in_batch, padded_answers_in_batch, \
                  seqlen_questions_in_batch, seqlen_answers_in_batch, emotions

    def size(self):
        return self._size

    def train_val_split(self, val_percent=1, random_split=True, move_samples=True):
        validate = DataBatch_db(self.db, self.input_vocabulary, self.output_vocabulary, self._ids[0: int(self._size * val_percent / 100)])
        train = DataBatch_db(self.db, self.input_vocabulary, self.output_vocabulary, self._ids[int(self._size * val_percent / 100):])
        return train, validate

    def sort(self):
        # not support
        pass

    def __del__(self):
        self._cursor.close()

    def _apply_padding(self, batch_of_sequences, vocabulary):
        """
        see same function in dataset.py
        """
        max_sequence_length = max([len(sequence) for sequence in batch_of_sequences])
        return [sequence + ([vocabulary.pad_int()] * (max_sequence_length - len(sequence))) for sequence in
                batch_of_sequences]
