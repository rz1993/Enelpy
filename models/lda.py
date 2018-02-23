'''
Simple Latent Dirichlet Allocation implementation with Collapsed Gibbs Sampling
for inference. Scipy's sparse matrices are used for efficient parameter storage
and updates.

Based off of Gregor Heinrich's excellent tutorial:
`Parameter estimation for text analysis`
'''
from collections import Counter
from scipy.sparse import csr_matrix, dok_matrix

import numpy as np
import random


def tokenize(line):
    return line.split()

def normalize(word):
    return word.lower()

class Docs:
    '''
    A simple document abstraction that builds and potentially reads from
    disk for large sets of files. The class exposes a document generator,
    each document of which is represented as a list of word ids.
    '''
    def __init__(self, fpaths=None):
        self.fpaths = []
        self._word2idx = {}
        self._docs = []
        if fpaths:
            self.read(fpaths)

    def read(self, fpaths):
        for fpath in fpaths:
            self.read_file(fpath)

    def read_file(self, fpath):
        word_id = len(self._word2idx) or 0
        doc = []
        with open(fpath, 'r') as f:
            for line in f:
                words = tokenize(line)
                words = map(normalize, words)
                for word in words:
                    if word not in self._word2idx:
                        self._word2idx[word] = word_id
                        word_id += 1
                    doc.append(self._word2idx[word])
        self._docs.append(doc)

    def write(self):
        pass

    def docs(self):
        for doc in self._docs:
            yield doc


class LDA:
    def __init__(self, ntopics, savesteps=100, alpha=1., beta=1., max_iter=1000):
        # Hyperparameters
        self._ntopics = ntopics
        self._alpha = alpha
        self._beta = beta

        # Parameters
        self._doc_topic = None
        self._topic_word = None
        self._topic_counts = None
        self._word2idx = {}

        # Configuration
        self._max_iter = max_iter
        self._savesteps = savesteps
        self._built = False

    def _build_params(self, docs):
        m, t, v = len(docs), self._ntopics, len(self._word2idx)
        self._doc_topic = dok_matrix((m, t), dtype=np.int8)
        self._topic_word = dok_matrix((t, v), dtype=np.int8)
        self._topic_counts = np.zeros(t)
        self._built = True

    def _add_count(self, doc_id, word_id, topic_id, count=1):
        self._doc_topic[doc_id, topic_id] += count
        self._topic_word[topic_id, word_id] += count
        self._topic_counts[topic_id] += count

    def get_doc_pdf(self, doc_id, norm=True):
        pdf = self._doc_topic[doc_id].toarray()
        if norm:
            pdf = pdf / np.sum(pdf)
        return pdf

    def get_topic_pdf(self, topic_id, norm=True):
        pdf = self._topic_word[topic_id, :].toarray()
        if norm:
            pdf = pdf / np.sum(pdf)
        return pdf

    def get_word_pdf(self, word_id, norm=True):
        pdf = self._topic_word[:, word_id].toarray()
        if norm:
            pdf = pdf / np.sum(pdf)
        return pdf

    def _init_assign(self, docs):
        assignments = []
        for d, doc in enumerate(docs.docs()):
            doc_assignments = []
            for word_id in doc:
                topic_id = random.randint(0, self._ntopics)
                self._add_count(d, word_id, topic_id)
                doc_assignments.append(topic_id)
            assignments.append(doc_assignments)
        return assignments

    def _build(self, docs):
        self._build_params(docs)

    def loglikelihood(self):
        '''
        Likelihood of a document:
            - p(w_mi|wdist_t)*p(z_mi=t|zdist_m)*p(zdist_m|alpha)*p(wdist_t|beta)
        Likelihood of a corpus is product of document likelihoods
        '''
        pass

    def converged(self):
        '''
        Compute the loglikelihood of the corpus. If the change has not exceeded
        some threshold then return True. Alternatively, also return True if
        the number of iterations exceeds self.max_iter.
        '''
        pass

    def fit(self, docs, verbose=False):
        if not self._built:
            self._build(docs)

        assignments = self._init_assign(docs)
        _iter = 1
        while not self.converged():
            for d, doc in enumerate(docs.docs()):
                doc_dist = self.get_doc_pdf(d)

                for i, word_id in enumerate(doc):
                    old_topic = assignments[d][i]
                    self._add_count(d, word_id,
                                    old_topic,
                                    count=-1)
                    topic_dist = (
                        self.get_word_pdf(word_id, norm=False) /
                        self._topic_counts *
                        self.get_doc_pdf(d))
                    new_topic = np.random.choice(self._ntopics,
                        p=topic_dist)
                    assignments[d][i] = new_topic
                    self._add_count(d, word_id,
                                    new_topic,
                                    count=1)

            if verbose and _iter % self._savesteps == 0:
                print('Iteration {0}: log likelihood {1:.4f}'.format(_iter, self.loglikelihood()))
            _iter += 1

    def transform(self, new_docs):
        pass
