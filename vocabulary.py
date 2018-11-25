"""
Vocabulary class
"""
import re
import io
import numpy as np

def read_embedding():
    fin = io.open('word_embed_clean.vec', 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = []
    word2index = {}
    index2word = {}
    for i, line in enumerate(fin):
        tokens = line.strip().split()
        data.append(list(map(float, tokens[1:])))
        assert len(data[-1]) == d
        word2index[tokens[0]] = i
        index2word[i] = tokens[0]
    fin.close()
    return word2index, index2word, np.asarray(data, dtype=np.float32), n, d

class Vocabulary(object):
    PAD = "<PAD>"
    SOS = "<SOS>"
    EOS = "<EOS>"
    OUT = "<OUT>"
    NUM = "<NUM>"
    special_tokens = [PAD, SOS, EOS, OUT, NUM]

    fname = "./datasets/embed_zhihu_clean.vec"
    
    def __init__(self):
        self.embedding = None
        self._word2count = {}
        self._words2int = {}
        self._ints2word = {}
        self._compiled = False

    def size(self):
        return len(self._words2int)
    
    def words2ints(self, words):
        return [self.word2int(w) for w in words.split()]
    
    def word2int(self, word):
        return self._words2int[word] if word in self._words2int else self.out_int()

    def ints2words(self, words_ints):
        words = ""
        for i in words_ints:
            word = self.int2word(i)
            if word not in ['.', '!', '?', '。', '！', '？']:
                words += " "
            words += word
        words = words.strip()
        return words

    def int2word(self, word_int):
        word = self._ints2word[word_int]       
        if word == 'i':
            word = 'I'
        return word
    
    def pad_int(self):
        return self.word2int(Vocabulary.PAD)

    def sos_int(self):
        return self.word2int(Vocabulary.SOS)
    
    def eos_int(self):
        return self.word2int(Vocabulary.EOS)

    def out_int(self):
        return self.word2int(Vocabulary.OUT)

    def num_int(self):
        return self.word2int(Vocabulary.NUM)

    
    @staticmethod        
    def load():
        vocabulary = Vocabulary()
        vocabulary.load_with_embedding()
        return vocabulary

    def load_with_embedding(self):
        fin = io.open(self.fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        n, d = map(int, fin.readline().split())
        data = []
        for i, line in enumerate(fin):
            tokens = line.strip().split()
            data.append(list(map(float, tokens[1:])))
            assert len(data[-1]) == d
            self._words2int[tokens[0]] = i
            self._ints2word[i] = tokens[0]
        self.embedding = np.asarray(data, dtype=np.float32)

    @staticmethod
    def clean_text(text, max_words = None):
        text = text.lower()
        text = re.sub(r"'+", "'", text)
        text = re.sub(r"i'm", "i am", text)
        text = re.sub(r"he's", "he is", text)
        text = re.sub(r"she's", "she is", text)
        text = re.sub(r"that's", "that is", text)
        text = re.sub(r"what's", "what is", text)
        text = re.sub(r"where's", "where is", text)
        text = re.sub(r"how's", "how is", text)
        text = re.sub(r"it's", "it is", text)
        text = re.sub(r"\'ll", " will", text)
        text = re.sub(r"\'ve", " have", text)
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"\'d", " would", text)
        text = re.sub(r"won't", "will not", text)
        text = re.sub(r"can't", "can not", text)
        text = re.sub(r"cannot", "can not", text)
        text = re.sub(r"n't", " not", text)
        text = re.sub(r"[()\"#/@;；:：<《>》{}`‘’'+=~|$&*%\[\]_]", " ", text)
        text = re.sub(r"[.]+", " . ", text)
        text = re.sub(r"[!]+", " ! ", text)
        text = re.sub(r"[?]+", " ? ", text)
        text = re.sub(r"[,-]+", " ", text)
        text = re.sub(r"[。]+", " . ", text)
        text = re.sub(r"[！]+", " ! ", text)
        text = re.sub(r"[？]+", " ? ", text)
        text = re.sub(r"[，-]+", " ", text)
        text = re.sub(r"[\t]+", " ", text)
        text = re.sub(r" +", " ", text)
        text = text.strip()
        
        #Truncate words beyond the limit, if provided.
        if max_words is not None:
            text_parts = text.split()
            if len(text_parts) > max_words:
                text = " ".join(text_parts[:max_words])

        if text == '':
            text = "?"
                
        return text
