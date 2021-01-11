import tensorflow as tf

# <s> and </s> are used in the data files to segment the abstracts into sentences. They don't receive vocab ids.
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'
PAD_TOKEN = '[PAD]'  # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNKNOWN_TOKEN = '[UNK]'  # This has a vocab id, which is used to represent out-of-vocabulary words
START_DECODING = '[START]'  # This has a vocab id, which is used at the start of every decoder input sequence
STOP_DECODING = '[STOP]'  # This has a vocab id, which is used at the end of untruncated target sequences

BAD_WORDS = [SENTENCE_START, SENTENCE_END, PAD_TOKEN, UNKNOWN_TOKEN, START_DECODING, STOP_DECODING]


def remove_bad_words(text):
    words = text.split()
    good_words = [word for word in words if word not in BAD_WORDS]
    return ' '.join(good_words)


def lens_to_time_step_masks(lens, max_len):
    masks = [[1 for i in range(l)] + [0 for i in range(max_len - l)] for l in lens]
    with tf.device('CPU'):
        masks = tf.constant(masks)
    return masks
