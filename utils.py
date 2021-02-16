import tensorflow as tf
import os
import numpy as np
from settings import batch_size


# <s> and </s> are used in the data files to segment the abstracts into sentences. They don't receive vocab ids.
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'
PAD_TOKEN = '[PAD]' # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNKNOWN_TOKEN = '[UNK]' # This has a vocab id, which is used to represent out-of-vocabulary words
START_DECODING = '[START]' # This has a vocab id, which is used at the start of every decoder input sequence
STOP_DECODING = '[STOP]' # This has a vocab id, which is used at the end of untruncated target sequences
BAD_WORDS = [SENTENCE_START, SENTENCE_END, PAD_TOKEN, UNKNOWN_TOKEN, START_DECODING, STOP_DECODING]


def remove_bad_words(text):
    words = text.split()
    good_words = [word for word in words if word not in BAD_WORDS]
    return ' '.join(good_words)


def lens_to_time_step_masks(lens, max_len):
    masks = [[1 for i in range(l)]+[0 for i in range(max_len-l)] for l in lens]
    with tf.device('CPU'):
        masks = tf.constant(masks)
    return masks


def save_model(model, full_path_to_checkpoints, epoch, batch_n):
    model.save_weights(os.path.join(full_path_to_checkpoints, 'pgn_epoch_' + str(epoch) + '_batch_' + str(batch_n)),
                       overwrite=True)


def save_loss(full_path_to_metrics, mean_epoch_loss, train_val='val'):
    f = open(os.path.join(full_path_to_metrics, train_val + '_ce_loss.txt'), 'a')
    f.write(str(mean_epoch_loss) + '\n')
    f.close()


def save_scores(full_path_to_metrics, scores, train_val='val'):
    f = open(os.path.join(full_path_to_metrics, train_val + '_bleurt.txt'), 'a')
    mean_score = np.mean(scores['bleurt'])
    f.write(str(mean_score) + '\n')
    f.close()
    f = open(os.path.join(full_path_to_metrics, train_val + '_r1.txt'), 'a')
    mean_score = np.mean(scores['1'])
    f.write(str(mean_score) + '\n')
    f.close()
    f = open(os.path.join(full_path_to_metrics, train_val + '_r2.txt'), 'a')
    mean_score = np.mean(scores['2'])
    f.write(str(mean_score) + '\n')
    f.close()
    f = open(os.path.join(full_path_to_metrics, train_val + '_rl.txt'), 'a')
    mean_score = np.mean(scores['l'])
    f.write(str(mean_score) + '\n')
    f.close()


def save_examples(full_path_to_examples, articles, gt_summaries, summaries, epoch, batch_n,
                  train_val='val', in_graph_decodings=False):
    new_dir_name = train_val + '_epoch_' + str(epoch) + '_batch_' + str(batch_n)
    path_to_dir = os.path.join(full_path_to_examples, new_dir_name)
    make_dir(path_to_dir)
    for i, n in enumerate(np.random.choice(len(articles), min(10, len(articles)), replace=False)):
        path_to_file = os.path.join(path_to_dir, 'example_' + str(i + 1) + '.txt')
        f = open(path_to_file, 'w')
        f.write(remove_bad_words(articles[n]))
        f.write('\n' + '#' * 50 + '\n')
        f.write(remove_bad_words(gt_summaries[n]))
        f.write('\n' + '#' * 50 + '\n')
        f.write(remove_bad_words(summaries[n]))
        if in_graph_decodings:
            f.write('\n' + '#' * 50 + '\n')
            f.write(in_graph_decodings[n])



def make_dir(path_to_dir):
    if not os.path.isdir(path_to_dir):
        os.mkdir(path_to_dir)


def make_dirs(path_to_checkpoints, experiment_name):
    make_dir(path_to_checkpoints)

    path_to_experiment = os.path.join(path_to_checkpoints, experiment_name)
    make_dir(path_to_experiment)

    model_checkpoints = os.path.join(path_to_experiment, 'model_checkpoints')
    examples_folder = os.path.join(path_to_experiment, 'examples')
    metrics_folder = os.path.join(path_to_experiment, 'metrics')

    make_dir(model_checkpoints)
    make_dir(examples_folder)
    make_dir(metrics_folder)

    return model_checkpoints, examples_folder, metrics_folder


def check_shapes(inputs):
    for tensor in inputs[-1].values:
        if tensor.shape[0] != batch_size:
            return False
    return True
