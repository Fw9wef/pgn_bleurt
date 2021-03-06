import tensorflow as tf
from tensorflow.keras import layers
from bleurt import score as bleurt_score
from utils import remove_bad_words, lens_to_time_step_masks
from utils import SENTENCE_START, SENTENCE_END, UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING
from rouge_score import rouge_scorer
from settings import bleurt_model, summary_max_tokens


class Detokenize(layers.Layer):
    """
    Этот класс реализует декодирование последовательности токенов в текст с использованием указателя (pointer)
    с поддержкой графового режима tf.
    """
    def __init__(self, vocab, **kwargs):
        super(Detokenize, self).__init__(**kwargs)
        self.vocab_size = vocab.size()
        self.end_id = vocab._word_to_id[STOP_DECODING]
        self.bad_words = [vocab._word_to_id[x] for x in [SENTENCE_START, SENTENCE_END, UNKNOWN_TOKEN,
                                                         PAD_TOKEN, START_DECODING, STOP_DECODING]]
        keys, values = [], []
        for key, value in vocab._id_to_word.items():
            keys.append(key); values.append(value)

        keys = tf.constant(keys, dtype=tf.int32)
        values = tf.constant(values)
        init = tf.lookup.KeyValueTensorInitializer(keys=keys, values=values)
        self.table = tf.lookup.StaticHashTable(init, default_value=UNKNOWN_TOKEN)

    def call(self, input_seqs, oovs):
        """
        Params:
            input_seqs: tf.Tensor: тензор с последовательностями токенов
            oovs: tf.Tensor: тензор с временным словарем с OOV словами
        """
        oovs_mask = tf.where(input_seqs > self.vocab_size, 1, 0)
        loss_mask = tf.where(input_seqs == self.end_id, 1, 0)
        loss_mask = tf.cumsum(loss_mask, axis=1, exclusive=True)
        loss_mask = tf.where(loss_mask == 0, 1, 0)
        loss_mask = tf.cast(loss_mask, tf.float32)

        bad_words = tf.TensorArray(dtype=tf.bool, size=0, dynamic_size=True)
        for i, bad_word_id in enumerate(self.bad_words):
            bad_words = bad_words.write(i, input_seqs == bad_word_id)
        bad_words = bad_words.stack()
        bad_words_mask = tf.math.reduce_any(bad_words, axis=0)

        # декодируем слова из словаря
        vocab_words = self.table.lookup(input_seqs)
        # декодируем слова из временного oov словаря
        oov_input_seqs = tf.where(input_seqs > self.vocab_size, input_seqs-self.vocab_size, 0)
        batch_inds = tf.expand_dims(tf.range(input_seqs.shape[0]), axis=-1)
        batch_inds = tf.tile(batch_inds, [1, summary_max_tokens+1])
        indices = tf.stack([batch_inds, oov_input_seqs], axis=-1)
        oov_words = tf.gather_nd(oovs, indices)

        words = tf.where(oovs_mask == 1, oov_words, vocab_words)
        words = tf.where(loss_mask == 0, b'', words)
        words = tf.where(bad_words_mask, b'', words)

        strings = tf.strings.reduce_join(words, axis=-1, separator=b' ')
        strings = tf.strings.strip(strings)
        return strings, loss_mask


class BleurtLayer(layers.Layer):
    """
    Этот класс создает слой, осуществляющий оценку резюме по метрике bleurt
    """
    def __init__(self, **kwargs):
        super(BleurtLayer, self).__init__(**kwargs)
        self.bleurt_ops = bleurt_score.create_bleurt_ops(bleurt_model)

    def call(self, gt_summary, pred_summary):
        """
        Params:
            gt_summary: tf.Tensor: тензор с гт резюме
            pred_summary: tf.Tensor: тензор со сгенерированными резюме
        """
        scores = self.bleurt_ops(gt_summary, pred_summary)
        return scores


class Env:
    """
    Класс инкапсулирет bleurt и rouge скореры. Также предоставляет методы для вычисления метрик.
    Методы не поддерживаются в графовом режиме!
    """
    def __init__(self, data, bleurt_device):
        self.data = data
        self.bleurt_device = bleurt_device
        with tf.device(self.bleurt_device):
            self.bleurt_scorer = bleurt_score.BleurtScorer(bleurt_model)
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def get_rewards(self, batch_texts, batch_tokens, batch_oovs, no_bad_words=True, get_rogue=True):
        """
        Метод выполняет вычисление метрик для батча резюме.
        Params:
            batch_texts: гт резюме
            batch_tokens: токены сгенерированного резюме
            batch_oovs: временные словари с oov ловами
            no_bad_words: bool: Если истина, то специальные слова ([UNK], [PAD], ...) удаляются из резюме, если ложь - остаются
            get_rogue: bool: Если истина, то выполняется подсчет rouge метрик, если ложь - только bleurt метрика
        """
        if no_bad_words:
            for i, text in enumerate(batch_texts):
                batch_texts[i] = remove_bad_words(text)

        summaries, summary_lens = [], []
        for i, (tokens, oovs) in enumerate(zip(batch_tokens, batch_oovs)):

            # для декодирования oov слов применяется метод detokenize класса Data. Метод не поддерживается в графовом режиме.
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


class CELoss(layers.Layer):
    """
    Класс представляет из себя слой перекрестной энтропии
    """
    def __init__(self, alpha=1., layer_name='ce_loss'):
        super(CELoss, self).__init__(name=layer_name)
        self.alpha = alpha

    def set_alpha(self, alpha):
        self.alpha = alpha

    def call(self, gt, probs, time_step_mask):
        """
        Params:
            gt: tf.Tensor: индексы гт токенов
            probs: tf.Tensor: распределение вероятностей генерируемого токена
            time_step_mask: tf.Tensor: маска пэддингов
        """
        # probs.shape = (batch, seqlen, classes)
        gt, probs, time_step_mask = gt[:, 1:], probs[:, :-1], time_step_mask[:, 1:]

        loss = tf.keras.losses.sparse_categorical_crossentropy(gt, probs, from_logits=False)
        mask = tf.math.logical_not(tf.math.equal(gt, 0))
        mask = tf.cast(mask, dtype=loss.dtype)
        loss = tf.math.reduce_sum(loss * mask, axis=1)
        n_samples = tf.math.reduce_sum(mask, axis=1)

        loss = loss / n_samples * self.alpha
        return loss


class RLLoss(layers.Layer):
    """
    Класс осуществляет ошибку при обучении с помощью алгоритма self-critic
    """
    def __init__(self, alpha=1., layer_name='rl_loss'):
        super(RLLoss, self).__init__(name=layer_name)
        self.alpha = alpha

    def set_alpha(self, alpha):
        self.alpha = alpha

    def call(self, chosen_tokens, probs, time_step_mask, delta_rewards):
        """
        Params:
            chosen_tokens: tf.Tensor: тензор со сгенерированными токенами (сэмплированными из распределения)
            probs: tf.Tensor: тензор с распределением вероятностей токенов на каждом шаге генерации
            time_step_mask: tf.Tensor: маска пэддингов
            delta_rewards: tf.Tensor: награды (разность наград)
        """
        batch_inds = tf.expand_dims(tf.range(probs.shape[0]), axis=1)
        batch_inds = tf.tile(batch_inds, [1, summary_max_tokens+1])

        time_inds = tf.expand_dims(tf.range(summary_max_tokens+1), axis=0)
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
