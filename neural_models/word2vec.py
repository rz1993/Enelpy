'''
Reimplementation of word2vec (Mikolov, et. al 2013):
https://arxiv.org/pdf/1301.3781.pdf

Intuitive blog post by Chris McCormick:
http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/

Features:
    - Skip-gram model
    - Negative sampling
    - Frequency-based phrase collocation
    - Stop word subsampling
'''
from collections import Counter

import tensorflow as tf


options = {}

class Options:
    pass

def generate_batch(options):
    pass


class Word2Vec:
    def __init__(self, window_size=5,
        subsample=0.001,
        neg_sample=5,
        hidden_dim=300):
        self._window_size = window_size
        self._subsample = sample
        self._neg_sample = neg_sample
        self._hdim = hidden_dim
        self._lr = 0.001
        self._opt = None

    def _build_train(self, sampling=True):
        batch_size = 1000
        vocab_size = 1000
        lr = self._lr
        window_size = self._window_size
        subsample = self._subsample
        neg_sample = self._neg_sample
        hdim = self._hdim

        self._inputs = tf.placeholder(tf.int32, shape=[batch_size])
        self._labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
        # Embedding layer
        self._embeddings = tf.Variable(tf.random_normal([vocab_size, hdim], 0., 1.))
        self._embed = tf.nn.embedding_lookup(self._embeddings, self._inputs)
        # Context layer
        self._out_w = tf.Variable(tf.random_normal([vocab_size, hdim]))
        self._out_b = tf.Variable(tf.zeros([vocab_size,]))

        # NOTE: Implement negative sampling via tensorflow candidate samplers
        if sampling:
            '''
            Under negative sampling scheme, updates are only propagated to
            the vectors for the true and negative labels. Thus, we dot only on
            the associated vectors for each sampled label, hence all the lookups.
            Each outputted logit is used to compute a sigmoid (rather than a
            softmax) loss. Finally, the same negative labels are used across
            the entire batch, so `tf.matmul` is used to produce multiple
            neg logits for each word input.
            '''
            # Get `neg_sample` negative words using a distorted unigram frequency
            _sampled, _, _ = tf.nn.fixed_unigram_candidate_sampler(
                true_classes=self._labels,
                num_true=1,
                num_sampled=neg_sample,
                unique=True,
                range_max=vocab_size,
                distortion=0.75)

            # true_w: [batch_size, hdim], true_b: [batch_size]
            true_w = tf.nn.embedding_lookup(self._out_w, self._labels)
            true_b = tf.nn.embedding_lookup(self._out_b, self._labels)

            # sampled_w: [neg_sample, hdim], sampled_b: [neg_sample]
            sampled_w = tf.nn.embedding_lookup(self._out_w, _sampled)
            sampled_b = tf.nn.embedding_lookup(self._out_b, _sampled)

            # Use `reduce_sum` and `multiply` to get a dot product
            # true_logits: [batch_size]
            true_logits = tf.reduce_sum(tf.multiply(true_w, self._embed), 1) + true_b

            sampled_b = tf.reshape(sampled_b, [neg_sample])
            smpl_logits = tf.matmul(self._embed,
                                    sampled_w,
                                    transpose_b=True) + sampled_b

            true_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(true_logits), logits=true_logits)
            neg_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(smpl_logits), logits=smpl_logits)

            self._train_loss = (tf.reduce_sum(true_loss) +
                tf.reduce_sum(neg_loss)) / batch_size
        else:
            logits = tf.matmul(self._out_w, self._embed) + self._out_b
            self._train_loss = tf.nn.sparse_softmax_cross_entropy(
                logits=logits, labels=self._labels)

        self._step = tf.train.AdamOptimizer(
            learning_rate=lr).minimize(self._train_loss)

    def _build_eval(self):
        pass

    def train(self, num_epochs=10, verbose=False):
        sess = self._sess

        init = tf.global_variables_initializer()
        sess.run(init)

        c = 0
        for e in range(1, num_epochs+1):
            for words, labels in generate_batch(options):
                c += 1
                feed = {self._inputs: words,
                        self._labels: labels}
                _, loss = sess.run([self._step, self._train_loss],
                                   feed_dict=feed)

                if verbose and c % 500 == 0:
                    print('Epoch %i iter %i loss: %:.3f' % (e, c, loss))

    def _word_counts(self, corpus, _normalize):
        # TODO: Refactor naive implementation
        generator = (_normalize(w) for doc in corpus for w in doc)
        counts = Counter(generator)

    def _collocate(self, words, counts):
        pass

    def _subsample(self, corpus, counts):
        pass

    def fit(self, corpus):
        pass
