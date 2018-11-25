from os import path
from dataset_reader import DatasetReader
import sqlite3
import numpy as np
import DataBatch_db
import vocabulary

class DataBaseReader(DatasetReader):

    def _get_dialog_lines_and_conversations(self, dataset_dir):
        pass

    def __init__(self):
        super(DataBaseReader, self).__init__('training_data_in_database')

    def read_dataset(self, dataset_dir, model_dir, training_hparams, share_vocab=True):
        if not share_vocab and training_hparams.conv_history_length > 0:
            raise ValueError("If training_hparams.conv_history_length > 0, share_vocab must be True since previous answers will be appended to the questions.")
        if share_vocab and training_hparams.input_vocab_threshold != training_hparams.output_vocab_threshold:
            raise ValueError("Cannot share vocabulary when the input and output vocab thresholds are different.")

        database_path = path.join(dataset_dir, 'xiaohuangji50w_nofenci_clean.sqlite3')

        db = sqlite3.connect(database_path)
        size = db.execute('SELECT count (*) AS num FROM conversation').fetchall()[0][0]
        print('open database, {} items altogether'.format(size))
        input_vocabulary = vocabulary.Vocabulary.load()
        embeddings = input_vocabulary.load_with_embedding()
        ids = range(size)
        databatch = DataBatch_db.DataBatch_db(db, input_vocabulary, input_vocabulary)
        return databatch, embeddings
