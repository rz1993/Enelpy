from collections import defaultdict

import logging
import random


class AveragedPerceptron:
    def __init__(self):
        self._weights = defaultdict(lambda : defaultdict(float))
        self._totals = defaultdict(lambda : defaultdict(float))
        self._tstamps = defaultdict(lambda : defaultdict(int))
        self._updates = 0

    def _update(self, c, f, v):
        updates_since = self._updates - self._tstamps[f][c]
        self._totals[f][c] += updates_since * self._weights[f][c]
        self._weights[f][c] += v
        self._tstamps[f][c] = self._updates

    def update(self, feats, c_true, c_pred):
        for f in feats:
            self._update(c_pred, f, -1.)
            self._update(c_true, f, 1.)
        self._updates += 1

    def predict(self, features):
        scores = defaultdict(float)

        for feature in features:
            if feature not in self._weights:
                continue
            weights = self._weights[feature]
            for klass, weight in weights.items():
                scores[klass] += weight
        return max(self._classes, key=lambda klass: (scores[klass], klass))

    def average_weights(self):
        '''
        Average the updated weight vectors over the number of updates.
        Totals and update counts are accumulated so we don't need to
        keep track of weight vector versions due to sparse updates.
        '''
        for feat, weights in self._weights.items():
            for klass, w in weights.items():
                total = self._totals[f][c]
                total += (self._updates - self._tstamps[f][c]) * w
                avged = round(total / float(self._updates), 3)
                weights[klass] = total


class PerceptronTagger:
    '''
    An POS tagger implementation using the Averaged Percepton model, which
    is more regularized than the standard perceptron.
    '''

    def __init__(self, seq):
        self._features = set()
        self._weights = dict()
        self._custom_features = []
        self.logger =
        self.model = AveragedPerceptron()

    def _add_feature(self, (name, func)):
        self._custom_features.append((name, func))

    def _normalize(self, word):
        return word.strip().lower()

    def _add(self, features, new_feat, value=''):
        '''Template for how to construct a feature lookup key.'''
        return features.add('_'.join((new_feat, value)))

    def _get_features(self, word, prev_tag, prev2_tag):
        '''Construct a vector of indicator template features.'''
        features = set()

        self._add(features, 'bias')
        self._add(features, 'i suffix', word[-3:])
        self._add(features, 'i prefix', word[:3])
        self._add(features, 'i-1 tag', prev_tag)
        self._add(features, 'i-2 tag', prev2_tag)

        for name, _get in self._custom_features:
            self._add(features, name, _get(word, context, i))

        return features

    def train(self, docs, epochs=10, verbose=False):
        '''
        Train tagger on a set of tagged documents.
        TODO: Abstract training process from perceptron update scheme.

        :param docs: A list of (words, tags) tuples.
        :epochs: The number of times we want to train over the whole corpus.
        :verbose: Whether or not to log periodically.
        '''
        for e in range(epochs):
            loss = 0
            n = 0
            for words, tags in docs:
                words = map(self._normalize, words)
                prev = self._START
                prev2 = self._START
                for i, word in enumerate(words):
                    x = self._get_features(word, words, prev, prev2)
                    t_pred = self.model.predict(x)
                    if t_pred != tags[i]:
                        loss += 1
                        self.model.update(x, tags[i], t_pred)
                    n += 1
                    prev2 = prev
                    prev = tags[i]
                    if verbose and n % 100 == 0:
                        self.logger.info('-Epoch:{0} iteration:{1} mean loss:{2:.3f}'.format(e, n, float(loss)/n))
            random.shuffle(docs)
        self.model.average_weights()

    def tag(self, docs):
        '''
        Tags a untagged corpus of preprocessed words.

        :params docs: an iterable of word iterables
        '''
        all_tags = []
        for words in docs:
            words = map(self._normalize, words)
            prev = self._START
            prev2 = self._START
            tags = []
            for word in words:
                x = self._get_features(word, words, prev, prev2)
                tag = self.model.predict(x)
                tags.append(tag)
            all_tags.append(tags)
        return all_tags
