import numpy as np
import tensorflow as tf

'''
Based on
`Neural Architectures for Named Entity Recognition` (Lample et. al 2016):
https://arxiv.org/pdf/1603.01360.pdf

TODO:
    1. Bi-LSTM-CRF NER with IOB encoding
    2. Inputs: word embeddings
    3. Layer 1: Bi-directional LSTM layer with outputs concatenated
    4. Layer 2: Fully connected layer projects to NER tags
    5. Layer 3: CRF Layer for computing loss against gold labels and decoding
        - Loss is the mean cross entropy against the gold tag sequence
        - This means we average across the -log-likelihood of each gold tag sequence
        - The likelihood of a sequence of tags is a softmax function on seq scores
        - The seq score is the sum of transition energies + sum of tag scores
        - The tag scores are tag output scores at each step by the LSTM
        - Boundary energies are also added.
        - This is essentially softmax cross entropy for sequences
        - Normalization tractable because linear chain CRFs transitions are defined on bi-tags
        - The transition matrix is not trained
        - At test time, decoding is done using transition matrix and LSTM outputs
'''

class NERTagger:
    def __init__(self, vocab_size, n_tags):
        self.vocab_size = vocab_size
        self.n_tags = n_tags

    def _build_embeddings(self):
        vocab_size = self.vocab_size
        embed_size = self.embed_size
        self.embedding_input = tf.placeholder(tf.float32, [vocab_size, embed_size])
        self.embeddings = tf.Variable(tf.random_normal([vocab_size, embed_size]),
                                      trainable=False)
        self.embeddings.assign(embedding_input)

        self.char_embedding_input = tf.placeholder(tf.float32, [None, embed_size])
        self.char_embeddings = tf.Variable(tf.random_normal([None, embed_size]),
                                           trainable=False)
        self.char_embeddings.assign(embedding_input)

    def _build_train(self, batch_size, timesteps, lstm_dim, lr):
        vocab_size = self.vocab_size
        embed_size = self.embed_size

        label_ids = tf.placeholder(tf.int32, [batch_size, timesteps])
        word_ids = tf.placeholder(tf.int32, [batch_size, timesteps])

        # Build character-based word embeddings
        char_ids = tf.placeholder(tf.int32, [batch_size, timesteps, None])
        chars = tf.nn.embedding_lookup(self.char_embeddings, char_ids)
        chars = tf.reshape(chars, [batch_size * timesteps, -1, embed_size])

        char_lstm_fw = tf.contrib.rnn.LSTMCell(lstm_dim, state_is_tuple=True)
        char_lstm_bw = tf.contrib.rnn.LSTMCell(lstm_dim, state_is_tuple=True)

        _, ((char_out_fw, _), (char_out_bw, _)) = tf.nn.bidirectional_dynamic_rnn(
                                                    cell_fw=char_lstm_fw,
                                                    cell_bw=char_lstm_bw,
                                                    inputs=chars,
                                                    sequence_length=word_lengths,
                                                    dtype=tf.float32)

        char_embeddings = tf.concat([char_out_fw, char_out_bw], axis=-1)
        char_embeddings = tf.reshape(char_embeddings, [-1, timesteps, 2 * embed_size])

        word_embeddings = tf.nn.embedding_lookup(self.embeddings, word_ids)
        embeddings = tf.concat([word_embeddings, char_embeddings],
                               axis=-1)

        lstm_fw = tf.contrib.rnn.LSTMCell(lstm_dim, state_is_tuple=True)
        lstm_bw = tf.contrib.rnn.LSTMCell(lstm_dim, state_is_tuple=True)

        (out_fw, out_bw), _ = tf.nn.bidirection_dynamic_rnn(cell_fw=lstm_fw,
                                cell_bw=lstm_bw,
                                inputs=embeddings,
                                sequence_length=seq_lengths,
                                dtype=tf.float32)

        contexts = tf.concat([out_fw, out_bw], axis=-1)
        '''
        Here the concatenated context outputs are passed through a dense layer to
        compute the un-normalized scores for each tag class at each timestep. Since
        the LSTM outputs are 3D tensors of shape [batch_size, timesteps, lstm_dim],
        we will reshape them so that the each timestep is treated as independent i.e.
        into the shape [batch_size * timesteps, lstm_dim]. This assumes that all
        sequence dependent information has been captured by the Bi-LSTM.
        '''
        contexts = tf.reshape(output, [-1, 2*lstm_dim])
        tag_energies = tf.layers.dense(inputs=contexts, units=n_tags)
        tag_energies = tf.reshape(tag_energies, [batch_size, timesteps, n_tags])

        '''
        Compute mean negative log-likelihood of gold labels with our current parameters,
        which is the softmax cross entropy of gold sequence. The normalization term
        over all sequences can be computed using the forward algorithm for a Linear
        Chain CRF.
        '''
        log_likelihood, transition = tf.contrib.crf.crf_loglikelihood(
                                        tag_energies,
                                        label_ids,
                                        seq_lengths)
        loss = tf.reduce_mean(-log_likelihood)

        optimizer = tf.train.AdamOptimizer(lr=lr)
        train_step = optimizer.minimize(loss)

        # Inputs
        self.word_ids = word_ids
        self.char_ids = char_ids
        self.label_ids = label_ids

        # Layers
        self.embeddings = embeddings
        self.lstm_fw = lstm_fw
        self.lstm_bw = lstm_bw
        self.tag_energies = tag_energies
        self.transition = transition
        self.loss = loss
        self.optimizer = optimizer
        self.train_step = train_step

    def _build_eval(self):
        '''
        Build decoding layer, which uses tensorflow's implementation of viterbi
        decoding to get optimal tag sequences based on Linear Chain CRF scoring.
        '''
        self.pred_tags, self.pred_score = tf.contrib.crf.crf_decode(
            self.tag_energies,
            self.transition)

    def load_pretrained_embeddings(self):
        pass

    def fit(self, docs,
            batch_size=200,
            epochs=50,
            lstm_dim=200,
            lr=0.001):
        pass
