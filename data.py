import os
import glob
import struct
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tensorflow.train import Example
from utils import SENTENCE_START, SENTENCE_END, UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING
from settings import vocab_file, vocab_size, data_folder, article_max_tokens, summary_max_tokens


class Vocab(object):
    """Vocabulary class for mapping between words and ids (integers)"""

    def __init__(self, vocab_file=vocab_file, max_size=vocab_size):
        """
        Creates a vocab of up to max_size words, reading from the vocab_file.
        If max_size is 0, reads the entire vocab file.

        Args:
            vocab_file: path to the vocab file, which is assumed to contain 
                        "<word> <frequency>" on each line, sorted with most frequent word first.
                        This code doesn't actually use the frequencies, though.
            max_size: integer. The maximum size of the resulting Vocabulary.
        """
        self._word_to_id = {}
        self._id_to_word = {}
        self._count = 0  # keeps track of total number of words in the Vocab

        # [PAD], <s>, </s>, [UNK], [START] and [STOP] get the ids 0,1,2,3,4,5.
        for w in [PAD_TOKEN, SENTENCE_START, SENTENCE_END, UNKNOWN_TOKEN, START_DECODING, STOP_DECODING]:
            self._word_to_id[w] = self._count
            self._id_to_word[self._count] = w
            self._count += 1

        self.UNKid = self.word2id(UNKNOWN_TOKEN)
        self.STOPid = self.word2id(STOP_DECODING)

        # Read the vocab file and add words up to max_size
        with open(vocab_file, 'r') as vocab_f:
            for line in vocab_f:
                pieces = line.split()
                if len(pieces) != 2:
                    print('Warning: incorrectly formatted line in vocabulary file: %s\n' % line)
                    continue
                w = pieces[0]
                if w in [SENTENCE_START, SENTENCE_END, UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
                    raise Exception(
                        '<s>, </s>, [UNK], [PAD], [START] and [STOP] shouldn\'t be in the vocab file, but %s is' % w)
                if w in self._word_to_id:
                    raise Exception('Duplicated word in vocabulary file: %s' % w)
                self._word_to_id[w] = self._count
                self._id_to_word[self._count] = w
                self._count += 1
                if max_size != 0 and self._count >= max_size:
                    print("max_size of vocab was specified as %i; we now have %i words. Stopping reading." % \
                          (max_size, self._count))
                    break

    def word2id(self, word):
        # Return word index in the vocabulary
        if word not in self._word_to_id:
            return self._word_to_id[UNKNOWN_TOKEN]
        return self._word_to_id[word]

    def id2word(self, word_id):
        # Return a word specified by the given index
        if word_id not in self._id_to_word:
            raise ValueError('Id not found in vocab: %d' % word_id)
        return self._id_to_word[word_id]

    def size(self):
        # Return vocabulary size
        return self._count


class Data:
    """
    Dataset class for creating generator object for tf.data.Dataset class from generator
    """
    def __init__(self, path_to_data=data_folder, mode='train' , vocab = None):
        """
        Creates a DataGen class object.

        Args:
            path_to_data: path to data chunks (see more https://github.com/abisee/cnn-dailymail)
            vocab: Vocab class object for performing tokenization
            mode: specifies train or test chunks will be loaded
        """
        assert mode in ['train', 'test', 'val'], 'invald DataGen mode'
        self.mode = mode
        if mode == 'train':
            self.list_of_files = [os.path.join(path_to_data, x) for x in os.listdir(path_to_data) \
                                  if ('train' in x and '.bin' in x)]
        elif mode == 'val':
            self.list_of_files = [os.path.join(path_to_data, x) for x in os.listdir(path_to_data) \
                                  if ('val' in x and '.bin' in x)]
        elif mode == 'test':
            self.list_of_files = [os.path.join(path_to_data, x) for x in os.listdir(path_to_data) \
                                  if ('test' in x and '.bin' in x)]

        if vocab:
            self.vocab = vocab
        else:
            self.vocab = Vocab()
        self.stop_decoding = self.vocab.word2id(STOP_DECODING)

    def get_tokenizer(self):
        vocab = self.vocab
        start_decoding = vocab.word2id(START_DECODING)
        stop_decoding = vocab.word2id(STOP_DECODING)
        unk_token = vocab.word2id(UNKNOWN_TOKEN)
        pad_token = vocab.word2id(PAD_TOKEN)

        def tokenizer(article, summary, max_article_len, max_summary_len):
            # tokenize article
            article_words = article.split()
            if len(article_words) > max_article_len:
                article_words = article_words[:max_article_len]
            tokenized_article_len = len(article_words)

            oovs_id = self.vocab.size()
            oovs2id, id2oovs = dict(), dict()
            tokenized_article = [start_decoding]
            extended_tokenized_article = [start_decoding]

            for w in article_words:
                if w not in [SENTENCE_START, SENTENCE_END, UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
                    tokenized_article.append(vocab.word2id(w))
                    if vocab.word2id(w) == vocab.UNKid:
                        if w in oovs2id.keys():
                            extended_tokenized_article.append(oovs2id[w])
                        else:
                            oovs2id[w] = oovs_id
                            id2oovs[oovs_id] = w
                            oovs_id += 1
                            extended_tokenized_article.append(oovs2id[w])
                    else:
                        extended_tokenized_article.append(vocab.word2id(w))

            extended_tokenized_article += [stop_decoding] + [pad_token for i in
                                                             range(max_article_len - len(article_words))]
            tokenized_article += [stop_decoding] + [pad_token for i in range(max_article_len - len(article_words))]

            # tokenize summary
            summary_words = summary.split()
            if len(summary_words) > max_summary_len:
                summary_words = summary_words[:max_summary_len]
            tokenized_summary_len = len(summary_words)

            tokenized_summary = [start_decoding]
            extended_tokenized_summary = [start_decoding]

            for w in summary_words:
                if w not in [SENTENCE_START, SENTENCE_END, UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
                    tokenized_summary.append(vocab.word2id(w))
                    if vocab.word2id(w) == vocab.UNKid:
                        if w in oovs2id.keys():
                            extended_tokenized_summary.append(oovs2id[w])
                        else:
                            extended_tokenized_summary.append(vocab.UNKid)
                    else:
                        extended_tokenized_summary.append(vocab.word2id(w))

            extended_tokenized_summary += [stop_decoding] + [pad_token for i in
                                                             range(max_summary_len - len(summary_words))]
            tokenized_summary += [stop_decoding] + [pad_token for i in range(max_summary_len - len(summary_words))]

            return extended_tokenized_article, extended_tokenized_summary, id2oovs

        return tokenizer

    def extract_text_data(self, tf_example):
        article = tf_example.features.feature['article'].bytes_list.value[0].decode("utf-8")
        summary = tf_example.features.feature['abstract'].bytes_list.value[0].decode("utf-8")
        return article, summary

    def get_generator(self, max_article_len=article_max_tokens, max_summary_len=summary_max_tokens):
        all_files = self.list_of_files
        text_tokenizer = self.get_tokenizer()

        def generator():
            np.random.shuffle(all_files)  ###############
            for file in tqdm(all_files):  ###############
                reader = open(file, 'rb')  ##############
                while True:
                    len_bytes = reader.read(8)
                    if not len_bytes: break
                    str_len = struct.unpack('q', len_bytes)[0]
                    example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
                    tf_example = Example.FromString(example_str)

                    article, summary = self.extract_text_data(tf_example)

                    extended_tokenized_article, extended_tokenized_summary, oovs = text_tokenizer(article, summary,
                                                                                                  max_article_len,
                                                                                                  max_summary_len)

                    extended_tokenized_article = tf.constant(extended_tokenized_article)
                    extended_tokenized_summary = tf.constant(extended_tokenized_summary)

                    yield {'article_text': article,
                           'extended_article_tokens': extended_tokenized_article,
                           'summary_text': summary,
                           'extended_summary_tokens': extended_tokenized_summary,
                           'oovs': oovs}

        return generator

    def detokenize(self, tokens_list, oovs_dict):
        # print(tokens_list.shape, np.max(tokens_list), np.argmax(tokens_list), oovs_dict.keys())
        list_of_words = []
        for t in tokens_list:

            if t < self.vocab.size():
                word = self.vocab.id2word(t)
                if word == STOP_DECODING:
                    break
                elif word in [PAD_TOKEN, START_DECODING]:
                    continue
            else:
                word = oovs_dict[t]

            list_of_words.append(word)

        text = ' '.join(list_of_words)
        return text

    def get_all_data(self, get_tensor_dict=False):
        article_text = []
        # article_tokens = []
        extended_article_tokens = []
        oovs = []
        summary_text = []
        # summary_tokens = []
        extended_summary_tokens = []

        data_gen = self.get_generator()
        with tf.device('CPU'):
            for instance in data_gen():
                article_text.append(instance['article_text'])
                # article_tokens.append(instance['article_tokens'])
                extended_article_tokens.append(instance['extended_article_tokens'])
                summary_text.append(instance['summary_text'])
                # summary_tokens.append(instance['summary_tokens'])
                extended_summary_tokens.append(instance['extended_summary_tokens'])
                oovs.append(instance['oovs'])

            # article_tokens = tf.stack(article_tokens, axis=0)
            extended_article_tokens = tf.stack(extended_article_tokens, axis=0)
            # summary_tokens = tf.stack(summary_tokens, axis=0)
            extended_summary_tokens = tf.stack(extended_summary_tokens, axis=0)

            # article_lens = tf.where(extended_article_tokens == self.stop_decoding)[:, 1]
            summary_lens = tf.where(extended_summary_tokens == self.stop_decoding)[:, 1]

            # article_tokens = article_tokens[:,:-1]
            extended_article_tokens = extended_article_tokens[:, :-1]
            # summary_tokens = summary_tokens[:,:-1]
            extended_summary_tokens = extended_summary_tokens[:, :-1]

            summary_max_len = extended_summary_tokens.shape[1]
            summary_loss_points = [[1 for i in range(x)] + [0 for i in range(summary_max_len - x)] for x in
                                   summary_lens]
            summary_loss_points = tf.constant(summary_loss_points, dtype=tf.float32)

            index = tf.constant([i for i in range(extended_article_tokens.shape[0])], dtype=tf.int32)
            index = tf.expand_dims(index, axis=1)

            # get oovs in tf.Tensor object
            max_oovs = 0
            for oov in oovs:
                curr_oov_size = len(oov)
                if curr_oov_size > max_oovs:
                    max_oovs = curr_oov_size

            tensor_oovs = []
            for oov in oovs:
                keys = list(oov.keys())
                keys.sort()
                oov_tensor = [oov[key] for key in keys] + [UNKNOWN_TOKEN for _ in range(max_oovs - len(keys))]
                oov_tensor = tf.constant(oov_tensor)
                tensor_oovs.append(oov_tensor)
            tensor_oovs = tf.stack(tensor_oovs, axis=0)


        return {'article_text': article_text,
                # 'article_tokens':article_tokens,
                'extended_article_tokens': extended_article_tokens,
                'summary_text': summary_text,
                # 'summary_tokens':summary_tokens,
                'extended_summary_tokens': extended_summary_tokens,
                'summary_loss_points': summary_loss_points,
                'index': index,
                'oovs': oovs,
                'tensor_oovs': tensor_oovs
                }
