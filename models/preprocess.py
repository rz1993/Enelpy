import itertools
import numpy as np
import re

from sklearn.externals import joblib


PAD = '<PAD>'
UNK = '<UNK>'

class TokenPreprocessor:
    def __init__(self,
                 char_indices=True,
                 lowercase=True,
                 norm_num=True,
                 padding=True,
                 vocab=None):
        self.char_indices = char_indices
        self.lowercase = lowercase
        self.norm_num = norm_num
        self.padding = padding
        self.vocab_chars = None
        self.vocab_tokens = vocab or {}

    def lower(self, word):
        return word.lower()

    def norm_number(self, word):
        if self.norm_num:
            return re.sub(r'[0-9]', r'0', word)
        else:
            return word

    def preprocess(self, word):
        word = self.lower(word)
        word = self.norm_number(word)
        return word

    def get_char_ids(self, word):
        chars = self.vocab_chars
        return [chars[c] if c in chars else chars[UNK] for c in word]

    def fit(self, X):
        chars = {PAD: 0, UNK: 1}
        words = {PAD: 0, UNK: 1}

        chained = itertools.chain(*X)
        chained = map(self.preprocess, chained)
        for word in set(chained):
            words[word] = len(words)

            if not self.char_indices:
                continue

            for c in word:
                if c not in chars:
                    chars[c] = len(chars)

        self.vocab_chars = chars
        self.vocab_tokens = words

    def transform(self, X):
        '''
        :params X: iterable of lists of words

        :returns
            If self.char_indices is True:
                A tuple of word ids and char ids
            A list of word id arrays
        '''
        chars = []
        words = []

        vocab_chars = self.vocab_chars
        vocab_words = self.vocab_tokens
        max_length = 0
        for doc in X:
            char_ids = []
            word_ids = []
            doc = map(self.preprocess, doc)
            for word in doc:
                if word in vocab_words:
                    word_id = vocab_words[word]
                else:
                    word_id = vocab_words[UNK]
                word_ids.append(word_id)

                if self.char_indices:
                    char_ids.append(self.get_char_ids(word))

            chars.append(char_ids)
            words.append(word_ids)

        result = words
        if self.char_indices:
            result = (words, chars)

        return result

    def save(self, filename):
        joblib.dump(self, filename)

    @classmethod
    def load(cls, filename):
        model = joblib.load(filename)
        return model
