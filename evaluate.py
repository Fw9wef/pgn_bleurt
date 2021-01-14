import os
import numpy as np
import tensorflow as tf
from data import Batcher, Data, Vocab
from model import PGN
from env import Env, CELoss, CoverLoss, RLLoss
from tqdm import tqdm
from utils import remove_bad_words

# import warnings
# warnings.filterwarnings('ignore')

# from silence_tensorflow import silence_tensorflow
# silence_tensorflow()


SUMMARY_MODEL_DEVICES = [i for i in range(4)]
BLEURT_DEVICES = [0]

pretrain_epochs = 15
batch_size = 4
global_batch_size = batch_size * len(SUMMARY_MODEL_DEVICES)

path_to_checkpoints = './checkpoints'
path_to_examples = './examples'
path_to_metrics = './metrics'
experiment_name = 'eval_slow_adam'

tf.debugging.set_log_device_placement(False)

# gpus = tf.config.experimental.list_physical_devices('GPU')
# gpus = [gpus[i] for i in SUMMARY_MODEL_DEVICES + BLEURT_DEVICES]
gpus = tf.config.experimental.list_physical_devices('GPU')
gpus = [gpus[2], gpus[3], gpus[4], gpus[5]]
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus, 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

summary_gpus = logical_gpus
print(logical_gpus)
bleurt_gpus = ['CPU']
# Define multigpu strategy
train_strategy = tf.distribute.MirroredStrategy(devices=summary_gpus)


def save_model(model, full_path_to_checkpoints, epoch, batch_n):
    model.save_weights(os.path.join(full_path_to_checkpoints, 'pgn_epoch_' + str(epoch) + '_batch_' + str(batch_n)),
                       overwrite=True)


def save_loss(full_path_to_metrics, mean_epoch_loss, train_val='val'):
    f = open(os.path.join(full_path_to_metrics, train_val + '_ce_loss.txt'), 'w')
    f.write(str(mean_epoch_loss) + '\n')
    f.close()


def save_scores(full_path_to_metrics, scores, train_val='val'):
    f = open(os.path.join(full_path_to_metrics, train_val + '_leurt.txt'), 'w')
    mean_score = np.mean(scores['bleurt'])
    f.write(str(mean_score) + '\n')
    f.close()
    f = open(os.path.join(full_path_to_metrics, train_val + '_r1.txt'), 'w')
    mean_score = np.mean(scores['1'])
    f.write(str(mean_score) + '\n')
    f.close()
    f = open(os.path.join(full_path_to_metrics, train_val + '_r2.txt'), 'w')
    mean_score = np.mean(scores['2'])
    f.write(str(mean_score) + '\n')
    f.close()
    f = open(os.path.join(full_path_to_metrics, train_val + '_rl.txt'), 'w')
    mean_score = np.mean(scores['l'])
    f.write(str(mean_score) + '\n')
    f.close()
    f = open(os.path.join(full_path_to_metrics, train_val + '_rw.txt'), 'w')
    mean_score = np.mean(scores['w'])
    f.write(str(mean_score) + '\n')
    f.close()


def save_examples(full_path_to_examples, articles, gt_summaries, summaries, epoch, batch_n, train_val='val'):
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


def make_dir(path_to_dir):
    if not os.path.isdir(path_to_dir):
        os.mkdir(path_to_dir)


def make_dirs(path_to_checkpoints, path_to_examples, path_to_metrics, experiment_name):
    make_dir(path_to_checkpoints)
    make_dir(path_to_examples)
    make_dir(path_to_metrics)

    full_path_to_checkpoints = os.path.join(path_to_checkpoints, experiment_name)
    full_path_to_examples = os.path.join(path_to_examples, experiment_name)
    full_path_to_metrics = os.path.join(path_to_metrics, experiment_name)

    make_dir(full_path_to_checkpoints)
    make_dir(full_path_to_examples)
    make_dir(full_path_to_metrics)

    return full_path_to_checkpoints, full_path_to_examples, full_path_to_metrics


def check_shapes(inputs):
    for tensor in inputs[-1].values:
        if tensor.shape[0] != batch_size:
            return False
    return True


vocab = Vocab()

##############################################################################################################################

# LOADING DATA

##############################################################################################################################
###Loading TRAIN data

###Loading Test data
test_data = Batcher(batch_size, data=Data(mode='test', vocab=vocab))
test_full_dataset = test_data.get_all_data()
test_article = test_full_dataset['article_text']
test_extended_input_tokens = test_full_dataset['extended_article_tokens']
test_summary = test_full_dataset['summary_text']
test_extended_gt_tokens = test_full_dataset['extended_summary_tokens']
test_index = test_full_dataset['index']
test_oovs = test_full_dataset['oovs']
test_loss_mask = test_full_dataset['summary_loss_points']

with tf.device('CPU'):
    test_tf_dataset = tf.data.Dataset.from_tensor_slices(
        (test_extended_input_tokens, test_extended_gt_tokens, test_loss_mask, test_index)).batch(int(global_batch_size))
    test_dist_dataset = train_strategy.experimental_distribute_dataset(test_tf_dataset)

max_oovs_in_text = max(0, np.max(test_extended_input_tokens) - vocab.size() + 1)
print('Max oovs in text :', max_oovs_in_text)

##############################################################################################################################

# DATA LOADED

##############################################################################################################################


##############################################################################################################################

# DEFINE MULTIGPU TRAIN STEP FUNCTIONS

##############################################################################################################################

with train_strategy.scope():
    path_to_model = './checkpoints/pretrain_mean_slowAdam/pgn_epoch_15_batch_last'
    model = PGN(vocab=vocab, max_oovs_in_text=max_oovs_in_text)
    model.load_weights(path_to_model)
    ce_loss = CELoss(alpha=1.)


def train_step(extended_input_tokens, extended_gt_tokens, loss_mask, idx):
    # extended_input_tokens, extended_gt_tokens, loss_mask, _ = inputs

    model.switch_decoding_mode('cross_entropy')

    with tf.GradientTape() as tape:
        gt_logits, greedy_seqs, coverage_losses = model(extended_input_tokens, extended_gt_tokens, training=True)
        loss = tf.nn.compute_average_loss(ce_loss(extended_gt_tokens, gt_logits, loss_mask),
                                          global_batch_size=global_batch_size)

    grads = tape.gradient(loss, model.trainable_weights)
    grads = [tf.clip_by_norm(g, 2) for g in grads]
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    return loss, greedy_seqs


def eval_step(extended_input_tokens, extended_gt_tokens, loss_mask, idx):
    model.switch_decoding_mode('self_critic')

    greedy_logits, sample_logits, greedy_seqs, sample_seqs, coverage_losses = model(extended_input_tokens,
                                                                                    extended_gt_tokens, training=False)
    loss = tf.nn.compute_average_loss(ce_loss(extended_gt_tokens, greedy_logits, loss_mask),
                                      global_batch_size=global_batch_size)

    return loss, greedy_seqs


@tf.function
def distributed_step(dist_inputs, mode):
    if mode == 'train':
        per_replica_losses, greedy_seqs = train_strategy.run(train_step, args=(dist_inputs))

    elif mode == 'eval':
        per_replica_losses, greedy_seqs = train_strategy.run(eval_step, args=(dist_inputs))

    return train_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None), greedy_seqs


##############################################################################################################################

# MULTIGPU TRAIN STEP FUNCTIONS DEFINED

##############################################################################################################################


env = Env(data=test_data.data, bleurt_device='CPU')

full_path_to_checkpoints, full_path_to_examples, full_path_to_metrics = make_dirs(path_to_checkpoints, path_to_examples,
                                                                                  path_to_metrics, experiment_name)

# no coverage loss training for first 15 epochs
total_batches = 0
test_batches_per_epoch = len(test_tf_dataset)

# tf.debugging.set_log_device_placement(True)


test_losses = []
test_sums = []
test_inds = []
test_iterator = iter(test_dist_dataset)
for test_batch_n in tqdm(range(test_batches_per_epoch)):
    batch = test_iterator.get_next()
    if not check_shapes(batch):
        continue
    loss, greedy_seqs = distributed_step(batch, 'eval')
    test_losses.append(loss)

    with tf.device('CPU'):
        test_sums += list(tf.concat(greedy_seqs.values, axis=0).numpy())
        test_inds += list(tf.concat(batch[-1].values, axis=0).numpy().squeeze())

articles = [test_article[x] for x in test_inds]
gt_summaries = [test_summary[x] for x in test_inds]
examples_oovs = [test_oovs[x] for x in test_inds]
scores, summaries, time_step_masks = env.get_rewards(gt_summaries, test_sums, examples_oovs)
# save_examples(full_path_to_examples, articles, gt_summaries, summaries, epoch, batch_n, 'val')
save_scores(full_path_to_metrics, scores, 'test')

mean_epoch_loss = np.mean(test_losses)
save_loss(full_path_to_metrics, mean_epoch_loss, 'test')

print("Testing complete:)")