import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_probability as tfp
# from contextvars import ContextVar


class Encoder(layers.Layer):
    def __init__(self, layer_name='encoder', lstm_units=128, vocab_size=10000,
                 embedding_dim=512, bahdanau_attention_units=128):
        super(Encoder, self).__init__(name=layer_name)
        self.lstm_units, self.embedding_dim, self.vocab_size = lstm_units, embedding_dim, vocab_size
        self.bahdanau_attention_units = bahdanau_attention_units

        self.enc_emb = layers.Embedding(vocab_size, embedding_dim, mask_zero=True, name="enc_emb")
        self.fw_lstm = layers.LSTM(lstm_units, return_sequences=True,
                                   return_state=True, name='fw_lstm')
        self.bw_lstm = layers.LSTM(lstm_units, return_sequences=True,
                                   return_state=True, go_backwards=True, name='bw_lstm')
        self.bidir_lstm = layers.Bidirectional(self.fw_lstm, backward_layer=self.bw_lstm, name='bidir_lstm')
        self.enc_state_layer = layers.Dense(bahdanau_attention_units, use_bias=False, name='enc_state_layer')

        self.state_m_compression = layers.Dense(lstm_units, activation='relu', name='state_m_compr')
        self.state_c_compression = layers.Dense(lstm_units, activation='relu', name='state_m_compr')

    def call(self, input_tokens):
        input_vectors = self.enc_emb(input_tokens)
        enc_output, fw_m_state, fw_c_state, bw_m_state, bw_c_state = self.bidir_lstm(input_vectors,
                                                                                     mask=input_vectors._keras_mask)

        dec_m_state = tf.concat([fw_m_state, bw_m_state], axis=1)
        dec_m_state = self.state_m_compression(dec_m_state)
        dec_c_state = tf.concat([fw_c_state, bw_c_state], axis=1)
        dec_c_state = self.state_c_compression(dec_c_state)

        enc_attn = self.enc_state_layer(enc_output)
        return [dec_m_state, dec_c_state], enc_output, enc_attn


class BahdanauAttention(layers.Layer):
    def __init__(self, layer_name='attention', bahdanau_attention_units=128):
        super(BahdanauAttention, self).__init__(name=layer_name)
        self.bahdanau_attention_units = bahdanau_attention_units

        self.decoder_state_layer = layers.Dense(bahdanau_attention_units, name='decoder_state_layer')
        self.coverage_state_layer = layers.Dense(bahdanau_attention_units, use_bias=False, name='coverage_state_layer')
        self.attn = layers.Dense(1, use_bias=False, name='attn')

    def call(self, enc_output, enc_attn, coverage_vector, dec_state):
        dec_attn = self.decoder_state_layer(dec_state)
        dec_attn = tf.expand_dims(dec_attn, 1)
        dec_attn = tf.repeat(dec_attn, repeats=enc_attn.shape[1], axis=1)

        cov_attn = tf.expand_dims(coverage_vector, -1)
        cov_attn = self.coverage_state_layer(cov_attn)

        attn = self.attn(tf.tanh(enc_attn + cov_attn + dec_attn))  # (bs, inp_l, 1)
        attn = tf.nn.softmax(attn, axis=1)
        c_vector = tf.math.reduce_sum(enc_output * attn, axis=1)
        return attn, c_vector


class DecodeStep(layers.Layer):
    def __init__(self, layer_name='decode_step', lstm_units=128, vocab_size=10000,
                 bahdanau_attention_units=128, gen_prob_units=128, max_oovs_in_text=100):
        super(DecodeStep, self).__init__(name=layer_name)
        self.lstm_units, self.vocab_size = lstm_units, vocab_size
        self.extend_distr = max_oovs_in_text

        self.attn_layer = BahdanauAttention(layer_name='attention',
                                            bahdanau_attention_units=bahdanau_attention_units)
        self.decoder_lstm = layers.LSTM(lstm_units, return_state=True, name='decoder_lstm')

        self.clf_layer = tf.keras.models.Sequential([
            layers.Dense(lstm_units, name='clf_layer_1'),
            layers.Dense(self.vocab_size, name='clf_layer_2'),
            layers.Softmax(name='clf_softmax', axis=-1)
        ], name='clf_layer')

        self.gen_prob_units = gen_prob_units
        self.gp_state_layer = layers.Dense(gen_prob_units, use_bias=False, name='gp_state_layer')
        self.gp_context_layer = layers.Dense(gen_prob_units, use_bias=False, name='gp_context_layer')
        self.gp_input_layer = layers.Dense(gen_prob_units, name='gp_input_layer')
        self.gen_prob_layer = layers.Dense(1, use_bias=False, name='gen_prob_layer')

    def call(self, extended_input_tokens, enc_output, enc_attn, dec_state, prev_word_vector, coverage_vector):
        # extend_distr = tf.maximum(0, tf.math.reduce_max(extended_input_tokens)-self.vocab_size+1)
        # extend_distr = 1

        dec_output, dec_m_state, dec_c_state = self.decoder_lstm(prev_word_vector, initial_state=dec_state)
        dec_state = [dec_m_state, dec_c_state]

        # computing attention, context vector
        attn, c_vector = self.attn_layer(enc_output, enc_attn, coverage_vector,
                                         tf.concat([dec_m_state, dec_c_state], axis=1))

        # computing coverage loss
        coverage_loss = tf.minimum(coverage_vector, attn[:, :, 0])
        coverage_loss = tf.math.reduce_sum(coverage_loss, axis=1)

        # adjusting coverage_vector
        coverage_vector += attn[:, :, 0]

        # computing probabilaty of generating next word from the vocabulary
        pg = tf.nn.sigmoid(self.gen_prob_layer(
            self.gp_state_layer(tf.concat([dec_m_state, dec_c_state], axis=1)) + \
            self.gp_context_layer(c_vector) + \
            self.gp_input_layer(prev_word_vector)
        )[0, 0, 0])

        # predict word from the vocabulary
        vocab_p = self.clf_layer(dec_output)
        base_distr_shape = vocab_p.shape  # (batch_size, vocab_size)
        vocab_p = tf.concat([vocab_p, tf.zeros((base_distr_shape[0], self.extend_distr))], axis=1)

        # predict word from the input text
        batch_ind = tf.expand_dims(tf.range(base_distr_shape[0]), 1)
        batch_ind = tf.tile(batch_ind, [1, extended_input_tokens.shape[1]])
        indices = tf.stack([batch_ind, extended_input_tokens], axis=-1)
        shape = tf.constant([base_distr_shape[0], self.vocab_size + int(self.extend_distr)])
        sentence_p = tf.scatter_nd(indices, attn[:, :, 0], shape)

        # summarize predictions of words from the vocabulary and from the input text
        final_distribution = pg * vocab_p + (1 - pg) * sentence_p
        # print(pg, tf.math.argmax(vocab_p, axis=1), tf.math.argmax(final_distribution, axis=1))

        return coverage_vector, final_distribution, dec_state, coverage_loss


class Decoder(layers.Layer):
    def __init__(self, decoding_mode='self_critic', layer_name='decoder', embedding_dim=512,
                 vocab=None, lstm_units=128, bahdanau_attention_units=128, gen_prob_units=128,
                 max_oovs_in_text=100):
        assert decoding_mode in ['self_critic', 'cross_entropy', 'evaluate'], 'Unknown decoding mode'
        self.decoding_mode = decoding_mode
        super(Decoder, self).__init__(name=layer_name)
        self.vocab, self.vocab_size, self.embedding_dim = vocab, vocab.size(), embedding_dim

        self.dec_emb = layers.Embedding(self.vocab_size, embedding_dim, name="dec_emb")
        self.decode_step = DecodeStep(layer_name='decode_step', lstm_units=lstm_units,
                                      vocab_size=self.vocab_size,
                                      bahdanau_attention_units=bahdanau_attention_units,
                                      gen_prob_units=gen_prob_units, max_oovs_in_text=max_oovs_in_text)

    def call(self, gt_tokens, extended_input_tokens, enc_output, enc_attn, rnn_state, tape=None):
        if self.decoding_mode in ['self_critic', 'evaluate']:
            greedy_coverage_vector = tf.zeros(extended_input_tokens.shape)
            sample_coverage_vector = tf.zeros(extended_input_tokens.shape)

            greedy_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
            sample_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
            greedy_seqs = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
            sample_seqs = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
            coverage_losses = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

            greedy_prev_word_vector = self.dec_emb(gt_tokens[:, :1])
            sample_prev_word_vector = greedy_prev_word_vector
            greedy_rnn_state, sample_rnn_state = rnn_state, rnn_state

            for i in range(gt_tokens.shape[1]):
                if self.decoding_mode == 'self_critic':
                    with tape.stop_recording():
                        greedy_output = self.decode_step(extended_input_tokens, enc_output, enc_attn,
                                                         greedy_rnn_state, greedy_prev_word_vector,
                                                         greedy_coverage_vector)
                        greedy_coverage_vector, greedy_pred_probs, greedy_rnn_state, _ = greedy_output
                else:
                    greedy_output = self.decode_step(extended_input_tokens, enc_output, enc_attn,
                                                     greedy_rnn_state, greedy_prev_word_vector,
                                                     greedy_coverage_vector)
                    greedy_coverage_vector, greedy_pred_probs, greedy_rnn_state, _ = greedy_output

                sample_output = self.decode_step(extended_input_tokens, enc_output, enc_attn,
                                                 sample_rnn_state, sample_prev_word_vector,
                                                 sample_coverage_vector)
                sample_coverage_vector, sample_pred_probs, sample_rnn_state, coverage_loss = sample_output

                greedy_token = tf.math.argmax(greedy_pred_probs, axis=1)
                greedy_token = tf.stop_gradient(greedy_token)
                greedy_token = tf.cast(greedy_token, tf.int32)

                sample_token = tfp.distributions.Categorical(probs=sample_pred_probs, dtype=tf.int32).sample()
                sample_token = tf.stop_gradient(sample_token)
                # sample_token = tf.cast(sample_token, tf.int32)

                greedy_probs = greedy_probs.write(i, greedy_pred_probs)
                sample_probs = sample_probs.write(i, sample_pred_probs)
                greedy_seqs = greedy_seqs.write(i, greedy_token)
                sample_seqs = sample_seqs.write(i, sample_token)
                coverage_losses = coverage_losses.write(i, coverage_loss)

                greedy_prev_word_vector = self.dec_emb(greedy_token)
                greedy_prev_word_vector = tf.expand_dims(greedy_prev_word_vector, axis=1)
                sample_prev_word_vector = self.dec_emb(sample_token)
                sample_prev_word_vector = tf.expand_dims(sample_prev_word_vector, axis=1)

            greedy_probs = tf.transpose(greedy_probs.stack(), [1, 0, 2])
            sample_probs = tf.transpose(sample_probs.stack(), [1, 0, 2])
            greedy_seqs = tf.transpose(greedy_seqs.stack(), [1, 0])
            sample_seqs = tf.transpose(sample_seqs.stack(), [1, 0])
            coverage_losses = tf.transpose(coverage_losses.stack(), [1, 0])

            return greedy_probs, sample_probs, greedy_seqs, sample_seqs, coverage_losses


        elif self.decoding_mode == 'cross_entropy':
            coverage_vector = tf.zeros(extended_input_tokens.shape)

            probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
            greedy_seqs = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
            coverage_losses = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

            gt_vectors = self.dec_emb(gt_tokens)

            gt_vectors = tf.transpose(gt_vectors, [1, 0, 2])
            for i in range(len(gt_vectors)):
                output = self.decode_step(extended_input_tokens, enc_output, enc_attn, rnn_state,
                                          tf.expand_dims(gt_vectors[i], 1), coverage_vector)
                coverage_vector, pred_probs, rnn_state, coverage_loss = output

                greedy_token = tf.math.argmax(pred_probs, axis=1)
                greedy_token = tf.cast(greedy_token, dtype=tf.int32)

                probs = probs.write(i, pred_probs)
                greedy_seqs = greedy_seqs.write(i, greedy_token)
                coverage_losses = coverage_losses.write(i, coverage_loss)

            probs = tf.transpose(probs.stack(), [1, 0, 2])
            greedy_seqs = tf.transpose(greedy_seqs.stack(), [1, 0])
            coverage_losses = tf.transpose(coverage_losses.stack(), [1, 0])

            return probs, greedy_seqs, coverage_losses


class PGN(tf.keras.models.Model):
    # class PGN(layers.Layer):
    def __init__(self, decoding_mode='self_critic', layer_name='pgn', embedding_dim=128,
                 vocab=None, lstm_units=256, bahdanau_attention_units=512, gen_prob_units=128,
                 max_oovs_in_text=100):
        assert decoding_mode in ['self_critic', 'cross_entropy', 'evaluate'], 'Unknown decoding mode'
        super(PGN, self).__init__(name=layer_name)
        self.vocab, self.decoding_mode = vocab, decoding_mode
        self.vocab_size = vocab.size()
        self.unk_token = vocab.UNKid

        self.encoder = Encoder(lstm_units=lstm_units, vocab_size=vocab.size(), embedding_dim=embedding_dim,
                               bahdanau_attention_units=bahdanau_attention_units)
        self.decoder = Decoder(decoding_mode=decoding_mode, embedding_dim=embedding_dim, vocab=vocab,
                               lstm_units=lstm_units, bahdanau_attention_units=bahdanau_attention_units,
                               gen_prob_units=gen_prob_units, max_oovs_in_text=max_oovs_in_text)

    def switch_decoding_mode(self, mode):
        assert mode in ['self_critic', 'cross_entropy', 'evaluate'], 'Unknown decoding mode'
        self.decoding_mode = mode
        self.decoder.decoding_mode = mode

    def call(self, extended_input_tokens, extended_gt_tokens, tape=None):

        input_tokens = tf.where(extended_input_tokens >= self.vocab_size, self.unk_token, extended_input_tokens)
        gt_tokens = tf.where(extended_gt_tokens >= self.vocab_size, self.unk_token, extended_gt_tokens)

        rnn_state, enc_output, enc_attn = self.encoder(input_tokens)

        if self.decoding_mode in ['self_critic', 'evaluate']:
            decoder_outputs = self.decoder(gt_tokens, extended_input_tokens, enc_output, enc_attn, rnn_state, tape=tape)
            greedy_probs, sample_probs, greedy_seqs, sample_seqs, coverage_losses = decoder_outputs
            return greedy_probs, sample_probs, greedy_seqs, sample_seqs, coverage_losses


        elif self.decoding_mode == 'cross_entropy':
            decoder_output = self.decoder(gt_tokens, extended_input_tokens, enc_output, enc_attn, rnn_state)
            gt_probs, greedy_seqs, coverage_losses = decoder_output
            return gt_probs, greedy_seqs, coverage_losses
