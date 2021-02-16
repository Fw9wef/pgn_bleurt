import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from bleurt import score as bleurt_score
from utils import remove_bad_words, lens_to_time_step_masks
from utils import SENTENCE_START, SENTENCE_END, UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING
from rouge_score import rouge_scorer
from settings import bleurt_model


class Detokenize(layers.Layer):
    def __init__(self, vocab, **kwargs):
        super(Detokenize, self).__init__(**kwargs)
        self.vocab_size = vocab.size()
        self.pad_id = vocab._word_to_id[PAD_TOKEN]
        self.end_id = vocab._word_to_id[SENTENCE_END]
        keys, values = [], []
        for key, value in vocab._id_to_word.items():
            keys.append(key); values.append(value)

        keys = tf.constant(keys, dtype=tf.int32)
        values = tf.constant(values)
        init = tf.lookup.KeyValueTensorInitializer(keys=keys, values=values)
        self.table = tf.lookup.StaticHashTable(init, default_value=UNKNOWN_TOKEN)

    def call(self, input_seqs, oovs):
        oovs_mask = tf.where(input_seqs>self.vocab_size, 1, 0)
        loss_mask = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        ones = tf.ones_like(input_seqs[0], dtype=tf.float32)
        zeros = tf.zeros_like(input_seqs[0], dtype=tf.float32)
        for sentence_n in range(input_seqs.shape[0]):
            ind_token_idx = tf.where(input_seqs[sentence_n] == self.end_id)[0]
            if ind_token_idx.shape[0] == 0:
                sentence_mask = ones
            else:
                end_token_idx = ind_token_idx[0, 0]
                sentence_mask = tf.concat([ones[:end_token_idx], zeros[:end_token_idx]], axis=0)
            loss_mask = loss_mask.write(sentence_n, sentence_mask)
        loss_mask = loss_mask.stack()

        # get vocab words
        vocab_words = self.table.lookup(input_seqs)
        # get oov words
        oov_input_seqs = tf.where(input_seqs>self.vocab_size, input_seqs-self.vocab_size, 0)
        batch_inds = tf.expand_dims(tf.range(input_seqs.shape[0]), axis=-1)
        batch_inds = tf.tile(batch_inds, [1, input_seqs.shape[1]])
        indices = tf.stack([batch_inds, oov_input_seqs], axis=-1)
        oov_words = tf.gather_nd(oovs, indices)

        words = tf.where(oovs_mask == 1, oov_words, vocab_words)
        words = tf.where(loss_mask == 0, b'', words)

        strings = tf.strings.join(words, separator=b' ')
        strings = tf.strings.strip(strings)
        return strings, loss_mask


class BleurtLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(BleurtLayer, self).__init__(**kwargs)
        self.bleurt_ops = bleurt_score.create_bleurt_ops(bleurt_model)

    def call(self, gt_summary, pred_summary):
        scores = self.bleurt_ops(gt_summary, pred_summary)
        return scores


class Env:
    def __init__(self, data, bleurt_device):
        self.data = data
        self.bleurt_device = bleurt_device
        with tf.device(self.bleurt_device):
            self.bleurt_scorer = bleurt_score.BleurtScorer(bleurt_model)
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def get_rewards(self, batch_texts, batch_tokens, batch_oovs, no_bad_words=True, get_rogue=True):

        if no_bad_words:
            for i, text in enumerate(batch_texts):
                batch_texts[i] = remove_bad_words(text)

        summaries, summary_lens = [], []
        for i, (tokens, oovs) in enumerate(zip(batch_tokens, batch_oovs)):
            summary = self.data.detokenize(tokens, oovs)
            summary_lens.append(len(summary.split()))
            if no_bad_words:
                summary = remove_bad_words(summary)
            summaries.append(summary)

        with tf.device(self.bleurt_device):
            bleurt_scores = self.bleurt_scorer.score(batch_texts, summaries)

        rouge_1_scores, rouge_2_scores, rouge_l_scores, rouge_w_scores = [], [], [], []
        if get_rogue:
            for target, prediction in zip(batch_texts, summaries):
                pair_score = self.rouge_scorer.score(target, prediction)
                rouge_1_scores.append(pair_score['rouge1'][2])
                rouge_2_scores.append(pair_score['rouge2'][2])
                rouge_l_scores.append(pair_score['rougeL'][2])

        scores = {
            'bleurt': bleurt_scores,
            '1': rouge_1_scores,
            '2': rouge_2_scores,
            'l': rouge_l_scores
        }

        time_step_masks = lens_to_time_step_masks(summary_lens, len(batch_tokens[0]))

        return scores, summaries, time_step_masks

    @tf.function
    def tf_get_rewards(self, gt_text, chosen_tokens, temp_oovs_tensor):
        bleurt_scores = 0
        return bleurt_scores


class CELoss(layers.Layer):
    def __init__(self, alpha=1., layer_name='ce_loss'):
        super(CELoss, self).__init__(name=layer_name)
        self.alpha = alpha

    def call(self, gt, probs, time_step_mask):
        # probs.shape = (batch, seqlen, classes)
        gt, probs, time_step_mask = gt[:, 1:], probs[:, :-1], time_step_mask[:, 1:]

        loss = tf.keras.losses.sparse_categorical_crossentropy(gt, probs, from_logits=False)
        mask = tf.math.logical_not(tf.math.equal(gt, 0))
        mask = tf.cast(mask, dtype=loss.dtype)
        loss = tf.math.reduce_sum(loss * mask, axis=1)
        n_samples = tf.math.reduce_sum(mask, axis=1)

        loss = loss / n_samples * self.alpha
        return loss


class CoverLoss(layers.Layer):
    def __init__(self, alpha=1., layer_name='cover_loss'):
        super(CoverLoss, self).__init__(name=layer_name)
        self.alpha = alpha

    def call(self, cover_loss, time_step_mask):
        return cover_loss * time_step_mask * self.alpha


class RLLoss(layers.Layer):
    def __init__(self, alpha=1., layer_name='rl_loss'):
        super(RLLoss, self).__init__(name=layer_name)
        self.alpha = alpha

    def call(self, chosen_tokens, probs, time_step_mask, delta_rewards):
        batch_inds = tf.expand_dims(tf.range(probs.shape[0]), axis=1)
        batch_inds = tf.tile(batch_inds, [1, probs.shape[1]])

        time_inds = tf.expand_dims(tf.range(probs.shape[1]), axis=0)
        time_inds = tf.tile(time_inds, [probs.shape[0], 1])

        inds = tf.stack([batch_inds, time_inds, chosen_tokens], axis=-1)
        # inds = tf.reshape(inds, (-1, 3))

        token_probs = tf.gather_nd(probs, inds)
        # token_probs = tf.reshape(token_probs, probs.shape[:-1])

        loss = - tf.math.log(token_probs) * time_step_mask
        loss = tf.math.reduce_sum(loss, axis=1) * delta_rewards
        n_tokens = tf.math.reduce_sum(time_step_mask, axis=1)
        mean_loss = loss / n_tokens * self.alpha
        return mean_loss
